# src/serialize.py
"""
Serialização de modelos para Mongo:
- Compat: base64 inline (antigo) -> serialize_tpot / carregar_modelo
- Recomendado: GridFS -> dump_model_to_gridfs / load_model_from_gridfs
"""

from __future__ import annotations

import base64
import gzip
import pickle
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pymongo.database import Database

try:
    # para load retrocompatível, alguns projetos usam dill
    import dill as dill_pickle  # type: ignore
except Exception:  # fallback
    dill_pickle = pickle  # type: ignore

# === Base64 (compatibilidade) =================================================


def serialize_tpot(modelo: Any) -> str:
    """
    Serializa o modelo via pickle e retorna string base64.
    Mantido por compatibilidade. Pode ultrapassar 16MB no Mongo.
    """
    data = pickle.dumps(modelo, protocol=pickle.HIGHEST_PROTOCOL)
    return base64.b64encode(data).decode("ascii")


def carregar_modelo(modelo_pickle_b64: str) -> Any:
    """
    Reconstroi o modelo a partir da string base64 gerada por serialize_tpot().
    Usa dill se disponível para maior compatibilidade.
    """
    if not isinstance(modelo_pickle_b64, str) or not modelo_pickle_b64.strip():
        raise ValueError("modelo_pickle_b64 inválido: forneça a string base64 do modelo.")
    data = base64.b64decode(modelo_pickle_b64)
    return dill_pickle.loads(data)


# === GridFS (recomendado) =====================================================

# Observação: estas funções não abrem conexão.
# Você deve passar `db` (pymongo.database.Database) já conectado.


def dump_model_to_gridfs(
    db: "Database",
    model: Any,
    *,
    bucket: str = "models",
    codec: str = "gzip",
    filename: str = "pipeline.pkl",
    extra_meta: Optional[dict] = None,
):
    """
    Serializa `model` e grava os bytes no GridFS.
    Retorna o ObjectId do arquivo no GridFS.

    Parâmetros:
    - db: pymongo.database.Database (conectado)
    - bucket: nome do bucket (gera <bucket>.files e <bucket>.chunks)
    - codec: "gzip" ou "none"
    - filename: nome lógico do arquivo (metadado)
    - extra_meta: dict opcional para enriquecer metadata (ex.: versões)

    Uso típico:
        fs_id = dump_model_to_gridfs(db, fitted, bucket="models", codec="gzip")
        doc["models"] = {"fitted_fs_id": fs_id, "bucket": "models", "codec": "gzip"}
    """
    from gridfs import GridFS  # import local para reduzir dependências globais

    raw = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    if codec and codec.lower() == "gzip":
        raw = gzip.compress(raw, compresslevel=9)
        stored_name = f"{filename}.gz" if not filename.endswith(".gz") else filename
    else:
        stored_name = filename

    meta = {"codec": codec or "none"}
    if extra_meta:
        try:
            meta.update(dict(extra_meta))
        except Exception:
            pass

    fs = GridFS(db, collection=bucket)
    file_id = fs.put(raw, filename=stored_name, metadata=meta)
    return file_id


def load_model_from_gridfs(
    db: "Database",
    file_id,
    *,
    bucket: str = "models",
) -> Any:
    """
    Lê bytes do GridFS pelo ObjectId `file_id` e reconstrói o modelo.
    Respeita metadata['codec']=="gzip" para descompressão.
    """
    from gridfs import GridFS  # import local

    fs = GridFS(db, collection=bucket)
    f = fs.get(file_id)  # pode aceitar ObjectId ou sua string correspondente
    data = f.read()

    codec = None
    try:
        codec = (f.metadata or {}).get("codec", "").lower()
    except Exception:
        codec = ""

    if codec == "gzip":
        data = gzip.decompress(data)

    return pickle.loads(data)


__all__ = [
    "serialize_tpot",
    "carregar_modelo",
    "dump_model_to_gridfs",
    "load_model_from_gridfs",
]