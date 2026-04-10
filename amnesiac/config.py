from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    embeddings_url: str = "http://localhost:7997"

    tg_api_id: int = 0
    tg_api_hash: str = ""
    tg_session_path: str = "session"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
