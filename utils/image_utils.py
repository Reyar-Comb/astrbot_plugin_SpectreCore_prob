import base64
import os
from typing import Iterable, Optional
from urllib.parse import unquote

from astrbot.api.all import Image, Reply, logger


class ImageUtils:
    """将 AstrBot 图片组件规范化为 Provider 可接受的 Base64 引用。"""

    @staticmethod
    def find_images(components: Optional[Iterable], _visited=None) -> list[Image]:
        """递归提取消息链及 ``Reply.chain`` 中的图片组件。"""
        if not components:
            return []

        visited = _visited if _visited is not None else set()
        images = []
        for component in components:
            component_id = id(component)
            if component_id in visited:
                continue
            visited.add(component_id)

            component_type = getattr(component, "type", None)
            class_name = component.__class__.__name__.lower()
            if isinstance(component, Image) or component_type == "image" or class_name == "image":
                images.append(component)
                continue

            if isinstance(component, Reply) or component_type == "reply" or class_name == "reply":
                images.extend(
                    ImageUtils.find_images(getattr(component, "chain", None), visited)
                )
        return images

    @staticmethod
    def find_reply_ids(components: Optional[Iterable]) -> list[str]:
        """提取当前消息链中的引用消息 ID，用于空 ``Reply.chain`` 回退。"""
        reply_ids = []
        for component in components or []:
            component_type = getattr(component, "type", None)
            class_name = component.__class__.__name__.lower()
            if isinstance(component, Reply) or component_type == "reply" or class_name == "reply":
                reply_id = str(getattr(component, "id", "") or "").strip()
                if reply_id and reply_id != "0" and reply_id not in reply_ids:
                    reply_ids.append(reply_id)
        return reply_ids

    @staticmethod
    def _existing_local_path(source: Optional[str]) -> Optional[str]:
        if not source:
            return None

        path = source
        if source.startswith("file:///"):
            path = unquote(source[8:])

        if os.path.isfile(path):
            return os.path.abspath(path)
        return None

    @staticmethod
    def _normalize_base64_ref(source: Optional[str]) -> Optional[str]:
        if not source:
            return None
        if source.startswith("base64://"):
            return source
        if source.startswith("data:image/") and ";base64," in source:
            return f"base64://{source.split(',', 1)[1]}"
        return None

    @staticmethod
    def _file_to_base64_ref(file_path: str) -> str:
        with open(file_path, "rb") as image_file:
            payload = base64.b64encode(image_file.read()).decode("ascii")
        return f"base64://{payload}"

    @staticmethod
    async def to_base64_ref(image: Image | str) -> Optional[str]:
        """
        将图片组件、URL 或本地路径转换为 ``base64://`` 引用。

        优先读取 AstrBot 本机上真实存在的持久化文件；否则交给
        ``Image.convert_to_base64()`` 通过组件的 URL 下载并编码。
        转换失败时返回 ``None``，避免把 NapCat 的裸文件名传给 Provider。
        """
        try:
            if isinstance(image, str):
                base64_ref = ImageUtils._normalize_base64_ref(image)
                if base64_ref:
                    return base64_ref

                local_path = ImageUtils._existing_local_path(image)
                if local_path:
                    return ImageUtils._file_to_base64_ref(local_path)

                image = Image(file=image)

            # 持久化后的 file 指向 AstrBot 本地文件，应优先于可能已过期的 url。
            local_path = ImageUtils._existing_local_path(getattr(image, "file", None))
            if local_path:
                return ImageUtils._file_to_base64_ref(local_path)

            for source in (getattr(image, "url", None), getattr(image, "file", None)):
                base64_ref = ImageUtils._normalize_base64_ref(source)
                if base64_ref:
                    return base64_ref

            payload = await image.convert_to_base64()
            if not payload:
                return None
            if payload.startswith("base64://"):
                return payload
            return f"base64://{payload}"
        except Exception as e:
            logger.warning(f"图片转 Base64 失败，已跳过该图片: {e}")
            return None
