from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    embeddings_url: str = "http://localhost:7997"

    tg_api_id: int = 0
    tg_api_hash: str = ""
    tg_session_path: str = "session"

    batch_size: int = 200
    inter_batch_sleep: float = 1.0

    sqlite_vec_enabled: bool = True

    proxy_host: str = ""
    proxy_port: int = 1080
    rag: dict = Field(default_factory=dict)

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
                proxy = data.get("proxy", {})
                if "proxy_host" not in self.model_fields_set:
                    self.proxy_host = proxy.get("host", self.proxy_host)
                if "proxy_port" not in self.model_fields_set:
                    self.proxy_port = proxy.get("port", self.proxy_port)
                if "rag" not in self.model_fields_set:
                    self.rag = data.get("rag", self.rag)
        except Exception:
            pass


settings = Settings()
