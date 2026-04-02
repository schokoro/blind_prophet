from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    embeddings_url: str = "http://localhost:7997"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
