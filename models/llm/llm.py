import json
import logging
from typing import Optional, Union, Generator, Any
from collections.abc import Mapping

from gigachat import GigaChat as GigaChatClient
from gigachat.models import Chat, Messages, MessagesRole, ChatCompletion, ChatCompletionChunk

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
    TextPromptMessageContent
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
    ) -> Union[LLMResult, Generator[LLMResultChunk]]:
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
        :return: full response or stream response chunk generator
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
    ) -> Union[LLMResult, Generator[LLMResultChunk]]:
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
        """Create GigaChat client with credentials"""
        client_params = {
            'credentials': credentials.get('api_key'),
            'verify_ssl_certs': credentials.get('verify_ssl_certs', True),
        }

        if credentials.get('scope'):
            client_params['scope'] = credentials['scope']

        if credentials.get('base_url'):
            client_params['base_url'] = credentials['base_url']

        return GigaChatClient(**client_params)

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
                messages.append(Messages(
                    role=MessagesRole.USER,
                    content=content,
                ))
            elif isinstance(message, AssistantPromptMessage):
                messages.append(Messages(
                    role=MessagesRole.ASSISTANT,
                    content=message.content,
                ))
                if message.tool_calls:
                    # Handle tool calls if present
                    for tool_call in message.tool_calls:
                        messages.append(Messages(
                            role=MessagesRole.FUNCTION,
                            content=json.dumps({
                                'name': tool_call.function.name,
                                'arguments': tool_call.function.arguments,
                            }),
                        ))
            elif isinstance(message, ToolPromptMessage):
                messages.append(Messages(
                    role=MessagesRole.FUNCTION,
                    content=json.dumps({
                        'name': message.name,
                        'content': message.content,
                    }),
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
                    # GigaChat supports images, but we'll handle them as text description for now
                    text_parts.append("[Image]")
            return ' '.join(text_parts)

        return ""

    def _convert_tools_to_functions(self, tools: list[PromptMessageTool]) -> list[dict]:
        """Convert Dify tools to GigaChat functions format"""
        functions = []
        for tool in tools:
            function = {
                'name': tool.name,
                'description': tool.description,
                'parameters': tool.parameters,
            }
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
        if hasattr(response.choices[0].message, 'function_call'):
            function_call = response.choices[0].message.function_call
            if function_call:
                assistant_message.tool_calls = [
                    AssistantPromptMessage.ToolCall(
                        id=f"call_{hash(function_call.name)}",
                        type="function",
                        function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                            name=function_call.name,
                            arguments=function_call.arguments,
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

                        # Add finish reason if present
                        if chunk.choices[0].finish_reason:
                            chunk_delta.finish_reason = chunk.choices[0].finish_reason

                            # Calculate usage for final chunk
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

    def get_num_tokens(
            self,
            model: str,
            credentials: dict,
            prompt_messages: list[PromptMessage],
            tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        """
        Get number of tokens for given prompt messages

        :param model: model name
        :param credentials: model credentials
        :param prompt_messages: prompt messages
        :param tools: tools for tool calling
        :return: number of tokens
        """
        # Convert messages to text for token counting
        prompt_text = ""
        for message in prompt_messages:
            content = self._extract_message_content(message)
            prompt_text += f"{content} "

        # Use GPT-2 tokenizer for estimation
        return self._get_num_tokens_by_gpt2(prompt_text)

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate model credentials

        :param model: model name
        :param credentials: model credentials
        """
        try:
            client = self._create_client(credentials)

            # Test with a simple ping message
            test_message = Messages(
                role=MessagesRole.USER,
                content="ping",
            )

            response = client.chat(Chat(
                messages=[test_message],
                model=model,
                max_tokens=5,
            ))

            if not response or not response.choices:
                raise CredentialsValidateFailedError("Invalid response from GigaChat API")

        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex))

    def _transform_invoke_error(self, error: Exception) -> InvokeError:
        """Transform GigaChat errors to Dify invoke errors"""
        error_message = str(error)

        if "401" in error_message or "Unauthorized" in error_message:
            return InvokeAuthorizationError(error_message)
        elif "429" in error_message or "rate limit" in error_message.lower():
            return InvokeRateLimitError(error_message)
        elif "400" in error_message or "Bad Request" in error_message:
            return InvokeBadRequestError(error_message)
        elif "500" in error_message or "502" in error_message or "503" in error_message:
            return InvokeServerUnavailableError(error_message)
        elif "connection" in error_message.lower() or "timeout" in error_message.lower():
            return InvokeConnectionError(error_message)
        else:
            return InvokeError(error_message)

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        """
        Map model invoke error to unified error
        """
        return {
            InvokeConnectionError: [ConnectionError, TimeoutError],
            InvokeServerUnavailableError: [Exception],  # Generic server errors
            InvokeRateLimitError: [],
            InvokeAuthorizationError: [],
            InvokeBadRequestError: [ValueError, TypeError],
        }
