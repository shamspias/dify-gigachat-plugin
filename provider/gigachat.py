import logging

from dify_plugin import ModelProvider
from dify_plugin.entities.model import ModelType
from dify_plugin.errors.model import CredentialsValidateFailedError

logger = logging.getLogger(__name__)


class GigaChatProvider(ModelProvider):
    def validate_provider_credentials(self, credentials: dict) -> None:
        """
        Validate provider credentials by making a real API call

        :param credentials: provider credentials
        """
        try:
            # Use the LLM model instance to validate
            model_instance = self.get_model_instance(ModelType.LLM)

            # Test with the basic GigaChat-2 model
            model_instance.validate_credentials(
                model="GigaChat-2",
                credentials=credentials
            )

            logger.info("GigaChat provider credentials validated successfully")

        except CredentialsValidateFailedError as ex:
            # Re-raise credential validation errors
            raise ex
        except Exception as ex:
            # Log and wrap other exceptions
            logger.exception(
                f"{self.get_provider_schema().provider} credentials validate failed"
            )
            raise CredentialsValidateFailedError(
                f"Failed to validate credentials: {str(ex)}"
            )
