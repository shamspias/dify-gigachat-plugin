import logging
from typing import Optional

from gigachat import GigaChat as GigaChatClient

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
        client = self._create_client(credentials)

        embeddings = []
        total_tokens = 0

        try:
            # GigaChat embeddings API processes texts one by one
            for text in texts:
                response = client.embeddings(texts=[text])

                if response and hasattr(response, 'data') and response.data:
                    embeddings.append(response.data[0].embedding)
                    # Estimate tokens if not provided
                    if hasattr(response, 'usage') and response.usage:
                        total_tokens += response.usage.total_tokens
                    else:
                        total_tokens += self._get_num_tokens_by_gpt2(text)
                else:
                    raise InvokeError("Invalid response from GigaChat embeddings API")

            # Calculate usage
            usage = self._calc_response_usage(
                model=model,
                credentials=credentials,
                tokens=total_tokens,
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

        :param model: model name
        :param credentials: model credentials
        :param texts: texts to embed
        :return: list of token counts
        """
        return [self._get_num_tokens_by_gpt2(text) for text in texts]

    def validate_credentials(
            self,
            model: str,
            credentials: dict,
    ) -> None:
        """
        Validate model credentials

        :param model: model name
        :param credentials: model credentials
        """
        try:
            client = self._create_client(credentials)

            # Test with a simple text
            response = client.embeddings(texts=["test"])

            if not response or not hasattr(response, 'data'):
                raise CredentialsValidateFailedError("Invalid response from GigaChat API")

        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex))

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

    def _calc_response_usage(
            self,
            model: str,
            credentials: dict,
            tokens: int,
    ) -> EmbeddingUsage:
        """
        Calculate response usage

        :param model: model name
        :param credentials: model credentials
        :param tokens: total tokens used
        :return: usage information
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
            latency=0,  # Will be calculated by framework
        )

        return usage

    def _transform_invoke_error(self, error: Exception) -> InvokeError:
        """Transform GigaChat errors to Dify invoke errors"""
        error_message = str(error)

        if "401" in error_message or "Unauthorized" in error_message:
            return InvokeError(f"Authentication failed: {error_message}")
        elif "429" in error_message or "rate limit" in error_message.lower():
            return InvokeError(f"Rate limit exceeded: {error_message}")
        elif "400" in error_message or "Bad Request" in error_message:
            return InvokeError(f"Invalid request: {error_message}")
        elif "500" in error_message or "502" in error_message or "503" in error_message:
            return InvokeError(f"Server error: {error_message}")
        elif "connection" in error_message.lower() or "timeout" in error_message.lower():
            return InvokeError(f"Connection error: {error_message}")
        else:
            return InvokeError(f"Unknown error: {error_message}")

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        """
        Map model invoke error to unified error
        """
        return {
            InvokeError: [Exception],  # Generic mapping
        }
