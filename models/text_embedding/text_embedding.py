import logging
import time
from typing import Optional

from gigachat import GigaChat as GigaChatClient
from gigachat.exceptions import AuthenticationError as GigaChatAuthError, ResponseError

from dify_plugin import TextEmbeddingModel
from dify_plugin.entities.model import EmbeddingInputType, PriceType
from dify_plugin.entities.model.text_embedding import (
    EmbeddingUsage,
    TextEmbeddingResult,
)
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeError,
)

logger = logging.getLogger(__name__)


class GigaChatTextEmbeddingModel(TextEmbeddingModel):
    """
    GigaChat Text Embedding Model implementation
    """

    # Model name mapping for consistency
    MODEL_MAPPING = {
        "GigaChat-Embeddings": "Embeddings",  # Map old naming convention
    }

    def _invoke(
            self,
            model: str,
            credentials: dict,
            texts: list[str],
            user: Optional[str] = None,
            input_type: EmbeddingInputType = EmbeddingInputType.DOCUMENT,
    ) -> TextEmbeddingResult:
        """
        Invoke text embedding model

        :param model: model name
        :param credentials: model credentials
        :param texts: texts to embed
        :param user: unique user id
        :param input_type: input type for embedding
        :return: embeddings result
        """
        # Map old model names to new ones for consistency
        actual_model = self.MODEL_MAPPING.get(model, model)

        client = self._create_client(credentials)

        embeddings = []
        total_tokens = 0
        start_time = time.time()

        try:
            # GigaChat processes all texts at once
            response = client.embeddings(texts=texts, model=actual_model)

            if response and hasattr(response, 'data') and response.data:
                for embedding_data in response.data:
                    embeddings.append(embedding_data.embedding)

                    # Get token usage if available
                    if hasattr(embedding_data, 'usage') and embedding_data.usage:
                        if hasattr(embedding_data.usage, 'prompt_tokens'):
                            total_tokens += embedding_data.usage.prompt_tokens
                    else:
                        # Estimate tokens if not provided
                        total_tokens += self._get_num_tokens_by_gpt2(texts[len(embeddings) - 1])
            else:
                raise InvokeError("Invalid response from GigaChat embeddings API")

            # Calculate usage
            usage = self._calc_response_usage(
                model=model,
                credentials=credentials,
                tokens=total_tokens,
                latency=time.time() - start_time,
            )

            return TextEmbeddingResult(
                embeddings=embeddings,
                usage=usage,
                model=model,
            )

        except Exception as e:
            raise self._transform_invoke_error(e)

    def get_num_tokens(
            self,
            model: str,
            credentials: dict,
            texts: list[str],
    ) -> list[int]:
        """
        Get number of tokens for given texts
        """
        # Map old model names to new ones for consistency
        actual_model = self.MODEL_MAPPING.get(model, model)

        # Try to use GigaChat's token counting
        try:
            client = self._create_client(credentials)
            result = client.tokens_count(input_=texts, model=actual_model)
            if result:
                return [token_info.tokens for token_info in result]
        except Exception as e:
            logger.debug(f"Failed to count tokens using GigaChat API: {e}")

        # Fallback to GPT-2 tokenizer
        return [self._get_num_tokens_by_gpt2(text) for text in texts]

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate credentials by making a real API call
        """
        # Map old model names to new ones for consistency
        actual_model = self.MODEL_MAPPING.get(model, model)

        try:
            # Try to generate embeddings for a test text
            test_text = ["тест"]

            client = self._create_client(credentials)
            response = client.embeddings(texts=test_text, model=actual_model)

            if not response or not hasattr(response, 'data') or not response.data:
                raise CredentialsValidateFailedError("Invalid response from API")

            logger.info("GigaChat embeddings credentials validated successfully")

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

    def _create_client(self, credentials: dict) -> GigaChatClient:
        """
        Build and return a configured GigaChatClient
        """
        # Convert verify_ssl_certs to boolean
        verify_ssl = self._get_verify_ssl_value(credentials)

        client_params = {
            "credentials": credentials.get("api_key"),
            "verify_ssl_certs": verify_ssl,
            "timeout": 30.0,
        }

        # Handle custom CA bundle
        import os
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

    def _calc_response_usage(
            self,
            model: str,
            credentials: dict,
            tokens: int,
            latency: float,
    ) -> EmbeddingUsage:
        """
        Calculate response usage
        """
        # Get input price info
        input_price_info = self.get_price(
            model=model,
            credentials=credentials,
            price_type=PriceType.INPUT,
            tokens=tokens,
        )

        # Create usage object
        usage = EmbeddingUsage(
            tokens=tokens,
            total_tokens=tokens,
            unit_price=input_price_info.unit_price,
            price_unit=input_price_info.unit,
            total_price=input_price_info.total_amount,
            currency=input_price_info.currency,
            latency=latency,
        )

        return usage

    def _transform_invoke_error(self, error: Exception) -> InvokeError:
        """Transform GigaChat errors to Dify invoke errors"""
        error_message = str(error)

        # Check for specific GigaChat exceptions
        if isinstance(error, GigaChatAuthError):
            return InvokeError(f"Authentication failed: {error_message}")

        if isinstance(error, ResponseError):
            # Parse status code from error
            if "401" in error_message or "403" in error_message:
                return InvokeError(f"Authentication error: {error_message}")
            elif "429" in error_message:
                return InvokeError(f"Rate limit exceeded: {error_message}")
            elif "400" in error_message:
                return InvokeError(f"Invalid request: {error_message}")
            elif "500" in error_message or "502" in error_message or "503" in error_message:
                return InvokeError(f"Server error: {error_message}")

        # Check error message patterns
        if "rate limit" in error_message.lower():
            return InvokeError(f"Rate limit exceeded: {error_message}")
        elif "connection" in error_message.lower() or "timeout" in error_message.lower():
            return InvokeError(f"Connection error: {error_message}")
        elif "ssl" in error_message.lower() or "certificate" in error_message.lower():
            return InvokeError(f"SSL/Certificate error: {error_message}")

        return InvokeError(f"Unknown error: {error_message}")

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        """
        Map model invoke error to unified error
        """
        return {
            InvokeError: [Exception],  # Generic mapping for embeddings
        }
