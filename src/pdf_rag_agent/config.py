from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    data_directory: Path
    chroma_directory: Path
    collection_name: str = "pdf_documents"

    @classmethod
    def from_project_root(
        cls,
        project_root: Path,
    ) -> "AppConfig":
        resolved_root = project_root.resolve()

        return cls(
            project_root=resolved_root,
            data_directory=resolved_root / "data",
            chroma_directory=resolved_root / "chroma_db",
        )

    def create_directories(self) -> None:
        self.data_directory.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.chroma_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

def get_default_config() -> AppConfig:
    return AppConfig.from_project_root(Path.cwd())