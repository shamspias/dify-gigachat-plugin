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

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        *Local* credential sanity‑check.

        We **do not** call the remote GigaChat API here; that could block for
        minutes and exceed the daemon's timeout.  Instead we validate the
        presence and basic syntax of the required fields.
        """
        # ----- API key --------------------------------------------------------
        api_key = credentials.get("api_key")
        if not api_key or not isinstance(api_key, str) or len(api_key) < 10:
            raise CredentialsValidateFailedError("API key looks invalid")

        # ----- scope ----------------------------------------------------------
        scope = credentials.get("scope")
        allowed_scopes = {
            "GIGACHAT_API_PERS",
            "GIGACHAT_API_B2B",
            "GIGACHAT_API_CORP",
        }
        if scope not in allowed_scopes:
            raise CredentialsValidateFailedError(
                f"Unknown scope value '{scope}'. Allowed: {', '.join(allowed_scopes)}"
            )

    def _create_client(self, credentials: dict) -> GigaChatClient:
        """
        Build and return a configured ``GigaChatClient``.

        ▸ Converts the form value ``verify_ssl_certs`` (which reaches us as the
          *string* ``"true"`` / ``"false"``) to a proper boolean.
        ▸ Sets a short ``request_timeout`` so a slow upstream cannot block the
          plugin‑daemon long enough to hit its 10‑minute watchdog.
        """
        # --- cast verify_ssl_certs to bool ------------------------------------
        verify_val = credentials.get("verify_ssl_certs", "false")
        if not isinstance(verify_val, bool):
            verify_val = str(verify_val).lower() == "true"

        client_params = {
            "credentials": credentials.get("api_key"),
            "verify_ssl_certs": verify_val,
            "request_timeout": 8,  # seconds
        }

        # Optional: honour a workspace‑level custom CA bundle
        # (set REQUESTS_CA_BUNDLE=/path/to/ca.pem)
        import os
        if not verify_val and os.getenv("REQUESTS_CA_BUNDLE"):
            client_params["verify_ssl_certs"] = os.environ["REQUESTS_CA_BUNDLE"]

        # optional extras ------------------------------------------------------
        if credentials.get("scope"):
            client_params["scope"] = credentials["scope"]

        if credentials.get("base_url"):
            client_params["base_url"] = credentials["base_url"]

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
