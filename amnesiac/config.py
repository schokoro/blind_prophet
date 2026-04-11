from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    embeddings_url: str = "http://localhost:7997"

    tg_api_id: int = 0
    tg_api_hash: str = ""
    tg_session_path: str = "session"

    batch_size: int = 200
    inter_batch_sleep: float = 1.0

    sqlite_vec_enabled: bool = True

    model_config = {"env_file": ".env", "extra": "ignore"}

    def model_post_init(self, __context):
        try:
            import yaml
            from pathlib import Path

            p = Path("config/params.yaml")
            if p.exists():
                data = yaml.safe_load(p.read_text())
                scraper = data.get("scraper", {})
                if "batch_size" not in self.model_fields_set:
                    self.batch_size = scraper.get("batch_size", self.batch_size)
                if "inter_batch_sleep" not in self.model_fields_set:
                    self.inter_batch_sleep = scraper.get("inter_batch_sleep", self.inter_batch_sleep)
        except Exception:
            pass


settings = Settings()
