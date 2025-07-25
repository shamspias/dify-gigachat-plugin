import json
import logging
import time
import os
from typing import Optional, Union, Generator, Any
from collections.abc import Mapping

from gigachat import GigaChat as GigaChatClient
from gigachat.models import Chat, Messages, MessagesRole, ChatCompletion, ChatCompletionChunk, \
    Function as GigaChatFunction
from gigachat.exceptions import AuthenticationError as GigaChatAuthError, ResponseError

from dify_plugin.entities.model.llm import (
    LLMResult,
    LLMResultChunk,
    LLMResultChunkDelta,
)
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    PromptMessage,
    PromptMessageTool,
    SystemPromptMessage,
    ToolPromptMessage,
    UserPromptMessage,
    ImagePromptMessageContent,
    TextPromptMessageContent,
    PromptMessageContent,
    PromptMessageContentType,
)
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)
from dify_plugin.interfaces.model.large_language_model import LargeLanguageModel

logger = logging.getLogger(__name__)


class GigaChatLargeLanguageModel(LargeLanguageModel):
    """
    GigaChat Large Language Model implementation
    """

    def _invoke(
            self,
            model: str,
            credentials: dict,
            prompt_messages: list[PromptMessage],
            model_parameters: dict,
            tools: Optional[list[PromptMessageTool]] = None,
            stop: Optional[list[str]] = None,
            stream: bool = True,
            user: Optional[str] = None,
    ) -> Union[LLMResult, Generator[LLMResultChunk, None, None]]:
        """
        Invoke large language model

        :param model: model name
        :param credentials: model credentials
        :param prompt_messages: prompt messages
        :param model_parameters: model parameters
        :param tools: tools for tool calling
        :param stop: stop sequences
        :param stream: is stream response
        :param user: unique user id
        :return: full response or stream response chunk generator result
        """
        return self._generate(
            model=model,
            credentials=credentials,
            prompt_messages=prompt_messages,
            model_parameters=model_parameters,
            tools=tools,
            stop=stop,
            stream=stream,
            user=user,
        )

    def _generate(
            self,
            model: str,
            credentials: dict,
            prompt_messages: list[PromptMessage],
            model_parameters: Mapping[str, Any],
            tools: Optional[list[PromptMessageTool]] = None,
            stop: Optional[list[str]] = None,
            stream: bool = True,
            user: Optional[str] = None,
    ) -> Union[LLMResult, Generator[LLMResultChunk, None, None]]:
        """
        Generate response using GigaChat API
        """
        client = self._create_client(credentials)

        # Convert prompt messages to GigaChat format
        messages = self._convert_prompt_messages_to_gigachat_messages(prompt_messages)

        # Prepare model parameters
        temperature = model_parameters.get('temperature', 0.7)
        max_tokens = model_parameters.get('max_tokens', 1024)
        top_p = model_parameters.get('top_p', 0.9)
        repetition_penalty = model_parameters.get('repetition_penalty', 1.0)

        # Prepare chat payload
        chat_payload = Chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stream=stream,
        )

        # Handle function calling if tools are provided
        if tools:
            chat_payload.functions = self._convert_tools_to_functions(tools)
            chat_payload.function_call = model_parameters.get('function_call', 'auto')

        try:
            if stream:
                return self._handle_stream_response(
                    client=client,
                    chat_payload=chat_payload,
                    model=model,
                    credentials=credentials,
                    prompt_messages=prompt_messages,
                )
            else:
                response = client.chat(chat_payload)
                return self._handle_response(
                    response=response,
                    model=model,
                    credentials=credentials,
                    prompt_messages=prompt_messages,
                )
        except Exception as e:
            raise self._transform_invoke_error(e)

    def _create_client(self, credentials: dict) -> GigaChatClient:
        """
        Build and return a configured GigaChatClient with proper error handling.
        """
        # Convert verify_ssl_certs to boolean
        verify_ssl = self._get_verify_ssl_value(credentials)

        client_params = {
            "credentials": credentials.get("api_key"),
            "verify_ssl_certs": verify_ssl,
            "timeout": 30.0,  # 30 seconds timeout
        }

        # Handle custom CA bundle
        if not verify_ssl and os.getenv("REQUESTS_CA_BUNDLE"):
            client_params["ca_bundle_file"] = os.environ["REQUESTS_CA_BUNDLE"]

        # Add optional parameters
        if credentials.get("scope"):
            client_params["scope"] = credentials["scope"]

        if credentials.get("base_url"):
            client_params["base_url"] = credentials["base_url"]

        return GigaChatClient(**client_params)

    def _get_verify_ssl_value(self, credentials: dict) -> bool:
        """Convert verify_ssl_certs to boolean"""
        verify_val = credentials.get("verify_ssl_certs", "false")
        if isinstance(verify_val, bool):
            return verify_val
        return str(verify_val).lower() == "true"

    def _convert_prompt_messages_to_gigachat_messages(
            self, prompt_messages: list[PromptMessage]
    ) -> list[Messages]:
        """Convert Dify prompt messages to GigaChat messages"""
        messages = []

        for message in prompt_messages:
            if isinstance(message, SystemPromptMessage):
                messages.append(Messages(
                    role=MessagesRole.SYSTEM,
                    content=self._extract_message_content(message),
                ))
            elif isinstance(message, UserPromptMessage):
                content = self._extract_message_content(message)
                attachments = self._extract_attachments(message)
                messages.append(Messages(
                    role=MessagesRole.USER,
                    content=content,
                    attachments=attachments if attachments else None,
                ))
            elif isinstance(message, AssistantPromptMessage):
                msg = Messages(
                    role=MessagesRole.ASSISTANT,
                    content=message.content or "",
                )

                # Handle function calls
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        if hasattr(tool_call.function, 'arguments'):
                            if isinstance(tool_call.function.arguments, str):
                                arguments = json.loads(tool_call.function.arguments)
                            else:
                                arguments = tool_call.function.arguments
                        else:
                            arguments = {}

                        msg.function_call = {
                            'name': tool_call.function.name,
                            'arguments': arguments,
                        }

                messages.append(msg)

            elif isinstance(message, ToolPromptMessage):
                messages.append(Messages(
                    role=MessagesRole.FUNCTION,
                    name=message.name,
                    content=message.content,
                ))

        return messages

    def _extract_message_content(self, message: PromptMessage) -> str:
        """Extract content from message, handling multimodal content"""
        if isinstance(message.content, str):
            return message.content

        if isinstance(message.content, list):
            text_parts = []
            for content in message.content:
                if isinstance(content, TextPromptMessageContent):
                    text_parts.append(content.data)
                elif isinstance(content, ImagePromptMessageContent):
                    # For now, we'll add a placeholder for images
                    # In the future, this could be enhanced to upload images
                    text_parts.append("[Изображение]")
            return ' '.join(text_parts)

        return ""

    def _extract_attachments(self, message: PromptMessage) -> Optional[list[str]]:
        """Extract attachment IDs from message content"""
        if not isinstance(message.content, list):
            return None

        attachments = []
        for content in message.content:
            if isinstance(content, ImagePromptMessageContent):
                # In a real implementation, you would upload the image
                # and get an attachment ID
                # For now, we'll skip this
                pass

        return attachments if attachments else None

    def _convert_tools_to_functions(self, tools: list[PromptMessageTool]) -> list[GigaChatFunction]:
        """Convert Dify tools to GigaChat functions format"""
        functions = []
        for tool in tools:
            function = GigaChatFunction(
                name=tool.name,
                description=tool.description,
                parameters={
                    'type': 'object',
                    'properties': tool.parameters.get('properties', {}),
                    'required': tool.parameters.get('required', []),
                }
            )
            functions.append(function)
        return functions

    def _handle_response(
            self,
            response: ChatCompletion,
            model: str,
            credentials: dict,
            prompt_messages: list[PromptMessage],
    ) -> LLMResult:
        """Handle non-streaming response"""
        # Extract response content
        assistant_message = AssistantPromptMessage(
            content=response.choices[0].message.content or ""
        )

        # Handle function calls if present
        if hasattr(response.choices[0].message, 'function_call') and response.choices[0].message.function_call:
            function_call = response.choices[0].message.function_call
            assistant_message.tool_calls = [
                AssistantPromptMessage.ToolCall(
                    id=f"call_{hash(function_call.name)}_{time.time_ns()}",
                    type="function",
                    function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                        name=function_call.name,
                        arguments=json.dumps(function_call.arguments) if isinstance(function_call.arguments,
                                                                                    dict) else function_call.arguments,
                    ),
                )
            ]

        # Calculate usage
        usage = self._calc_response_usage(
            model=model,
            credentials=credentials,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
        )

        return LLMResult(
            model=model,
            prompt_messages=prompt_messages,
            message=assistant_message,
            usage=usage,
        )

    def _handle_stream_response(
            self,
            client: GigaChatClient,
            chat_payload: Chat,
            model: str,
            credentials: dict,
            prompt_messages: list[PromptMessage],
    ) -> Generator[LLMResultChunk, None, None]:
        """Handle streaming response"""
        index = 0
        full_content = ""
        prompt_tokens = 0
        completion_tokens = 0

        try:
            for chunk in client.stream(chat_payload):
                if isinstance(chunk, ChatCompletionChunk):
                    if chunk.choices and chunk.choices[0].delta:
                        delta = chunk.choices[0].delta
                        if delta.content:
                            full_content += delta.content

                            # Create chunk delta
                            chunk_delta = LLMResultChunkDelta(
                                index=index,
                                message=AssistantPromptMessage(content=delta.content),
                            )

                            # Handle finish reason and usage
                            if chunk.choices[0].finish_reason:
                                chunk_delta.finish_reason = chunk.choices[0].finish_reason

                                # Get usage information
                                if hasattr(chunk, 'usage') and chunk.usage:
                                    prompt_tokens = chunk.usage.prompt_tokens or prompt_tokens
                                    completion_tokens = chunk.usage.completion_tokens or completion_tokens
                                else:
                                    # Estimate tokens if not provided
                                    prompt_tokens = self.get_num_tokens(
                                        model=model,
                                        credentials=credentials,
                                        prompt_messages=prompt_messages,
                                    )
                                    completion_tokens = self.get_num_tokens(
                                        model=model,
                                        credentials=credentials,
                                        prompt_messages=[AssistantPromptMessage(content=full_content)],
                                    )

                                chunk_delta.usage = self._calc_response_usage(
                                    model=model,
                                    credentials=credentials,
                                    prompt_tokens=prompt_tokens,
                                    completion_tokens=completion_tokens,
                                )

                            yield LLMResultChunk(
                                model=model,
                                prompt_messages=prompt_messages,
                                delta=chunk_delta,
                            )

                            index += 1

        except Exception as e:
            raise self._transform_invoke_error(e)

    def get_num_tokens(
            self,
            model: str,
            credentials: dict,
            prompt_messages: list[PromptMessage],
            tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        """
        Get number of tokens for given prompt messages
        """
        # GigaChat has a tokens_count method, let's try to use it
        try:
            client = self._create_client(credentials)

            # Convert messages to text
            prompt_text = ""
            for message in prompt_messages:
                content = self._extract_message_content(message)
                prompt_text += f"{content} "

            # Try to use GigaChat's token counting
            result = client.tokens_count(input_=[prompt_text], model=model)
            if result and len(result) > 0:
                return result[0].tokens
        except Exception as e:
            logger.debug(f"Failed to count tokens using GigaChat API: {e}")

        # Fallback to GPT-2 tokenizer
        prompt_text = ""
        for message in prompt_messages:
            content = self._extract_message_content(message)
            prompt_text += f"{content} "

        return self._get_num_tokens_by_gpt2(prompt_text)

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate credentials by making a real API call
        """
        try:
            # Create a test message
            test_message = UserPromptMessage(content="тест")

            # Try to generate a response
            response = self._generate(
                model=model,
                credentials=credentials,
                prompt_messages=[test_message],
                model_parameters={"max_tokens": 5, "temperature": 0.1},
                stream=False,
            )

            # If we get here, credentials are valid
            logger.info("GigaChat credentials validated successfully")

        except GigaChatAuthError as e:
            raise CredentialsValidateFailedError(f"Invalid API key: {str(e)}")
        except ResponseError as e:
            if "401" in str(e) or "403" in str(e):
                raise CredentialsValidateFailedError(f"Authentication failed: {str(e)}")
            elif "scope" in str(e).lower():
                raise CredentialsValidateFailedError(
                    f"Invalid scope for your account type. Please check your scope setting: {str(e)}"
                )
            else:
                raise CredentialsValidateFailedError(f"API error: {str(e)}")
        except Exception as e:
            error_msg = str(e)
            if "ssl" in error_msg.lower() or "certificate" in error_msg.lower():
                raise CredentialsValidateFailedError(
                    "SSL certificate error. Please check your SSL settings or disable SSL verification."
                )
            raise CredentialsValidateFailedError(f"Failed to validate credentials: {error_msg}")

    def _transform_invoke_error(self, error: Exception) -> InvokeError:
        """Transform GigaChat errors to Dify invoke errors"""
        error_message = str(error)

        # Check for specific GigaChat exceptions
        if isinstance(error, GigaChatAuthError):
            return InvokeAuthorizationError(f"Authentication failed: {error_message}")

        if isinstance(error, ResponseError):
            # Parse status code from error
            if "401" in error_message or "403" in error_message:
                return InvokeAuthorizationError(error_message)
            elif "429" in error_message:
                return InvokeRateLimitError(error_message)
            elif "400" in error_message:
                return InvokeBadRequestError(error_message)
            elif "500" in error_message or "502" in error_message or "503" in error_message:
                return InvokeServerUnavailableError(error_message)

        # Check error message patterns
        if "rate limit" in error_message.lower():
            return InvokeRateLimitError(error_message)
        elif "connection" in error_message.lower() or "timeout" in error_message.lower():
            return InvokeConnectionError(error_message)
        elif "ssl" in error_message.lower() or "certificate" in error_message.lower():
            return InvokeConnectionError(f"SSL/Certificate error: {error_message}")

        # Default error
        return InvokeError(error_message)

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        """
        Map model invoke error to unified error
        """
        return {
            InvokeConnectionError: [ConnectionError, TimeoutError],
            InvokeServerUnavailableError: [ResponseError],
            InvokeRateLimitError: [],
            InvokeAuthorizationError: [GigaChatAuthError],
            InvokeBadRequestError: [ValueError, TypeError],
        }
