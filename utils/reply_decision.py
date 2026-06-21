from astrbot.api.all import *
from typing import Dict, Any, List, Optional, Tuple
import random
import time
from .llm_utils import LLMUtils

class ReplyDecision:
    """
    消息回复决策工具类
    用于判断是否要使用大模型回复消息
    """

    _wakeup_states: Dict[str, Dict[str, Any]] = {}
    _recent_messages: Dict[str, List[Tuple[float, str]]] = {}
    _recent_reply_times: Dict[str, List[float]] = {}
    _last_bot_messages: Dict[str, Dict[str, Any]] = {}
    _user_interactions: Dict[str, Dict[str, Dict[str, Any]]] = {}
    
    @staticmethod
    def should_reply(event: AstrMessageEvent, config: AstrBotConfig) -> bool:
        """
        判断是否应该回复消息
        
        Args:
            event: 消息事件
            config: 配置对象
            
        Returns:
            是否应该回复
        """
        try:
            # 获取必要信息
            platform_name = event.get_platform_name()
            is_private_chat = event.is_private_chat()
            chat_id = event.get_sender_id() if is_private_chat else event.get_group_id()
            
            # 检查是否已有大模型在处理
            if LLMUtils.is_llm_in_progress(platform_name, is_private_chat, chat_id):
                logger.debug(f"当前聊天已有大模型处理中，不进行回复")
                return False
            
            # 检查是否处于临时静默状态
            mute_info = config.get("_temp_mute", {})
            if mute_info and mute_info.get("until", 0) > time.time():
                logger.debug(f"当前处于临时静默状态，不进行回复")
                return False
                
            # 检查消息是否包含黑名单关键词
            blacklist_keywords = config.get("model_frequency", {}).get("blacklist_keywords", [])
            if blacklist_keywords and ReplyDecision._check_blacklist_keywords(event, blacklist_keywords):
                logger.debug("消息中包含黑名单关键词，不进行回复")
                return False
            
            # 检查配置中的回复规则
            return ReplyDecision._check_reply_rules(event, config)
        except Exception as e:
            logger.error(f"判断是否回复时发生错误: {e}")
            return False
        finally:
            try:
                ReplyDecision._remember_user_message(event)
            except Exception as e:
                logger.debug(f"记录轻量消息状态失败: {e}")
    
    @staticmethod
    def _check_reply_rules(event: AstrMessageEvent, config: AstrBotConfig) -> bool:
        """
        检查回复规则
        
        Args:
            event: 消息事件
            config: 配置对象
            
        Returns:
            是否应该回复
        """
        # 检查是否是开启回复的群聊/私聊
        if event.is_private_chat():
            if not config.get("enabled_private", False):
                logger.debug("未开启私聊回复功能")
                return False
        else:
            group_id = event.get_group_id()
            if not group_id:
                logger.debug("群聊ID为空，不进行回复")
                return False
            group_id = str(group_id).strip()

            # 获取配置集合并规范化类型 (O(1) 查找)
            blocked_groups = {str(g).strip() for g in config.get("blocked_groups", []) if str(g).strip()}
            enabled_groups = {str(g).strip() for g in config.get("enabled_groups", []) if str(g).strip()}

            # 1. 黑名单检查 - 最高优先级
            if group_id in blocked_groups:
                logger.debug(f"群聊{group_id}在黑名单中，不进行回复")
                return False

            # 2. 全局开关检查
            if config.get("enable_all_groups", False):
                logger.debug(f"全局群聊回复已开启，允许回复群聊{group_id}")
                # 继续执行下面的频率检查
            elif group_id not in enabled_groups:
                # 3. 白名单检查 (仅在全局开关关闭时)
                logger.debug(f"群聊{group_id}未在白名单中，不进行回复")
                return False
            
        # 获取消息频率配置
        frequency_config = config.get("model_frequency", {})
        # 获取回复方法
        method = frequency_config.get("method", "概率回复")

        if method == "智能概率回复":
            return ReplyDecision._check_smart_probability(event, config, frequency_config)

        # 检查关键词触发
        keywords = frequency_config.get("keywords", [])
        if keywords and ReplyDecision._check_keywords(event, keywords):
            logger.debug("消息中包含关键词，触发回复")
            return True

        # 根据不同方法判断
        if method == "概率回复":
            prob_config = frequency_config.get("probability", {})
            
            # 私聊固定概率为1，群聊使用配置概率
            if event.is_private_chat():
                probability = 1.0  # 私聊总是回复
                logger.debug("私聊消息，固定概率为1，总是回复")
            else:
                probability = prob_config.get("probability", 0.1)  # 群聊使用配置概率
                logger.debug(f"群聊消息，使用配置概率: {probability}")
            
            # 使用概率计算是否回复
            should_reply = random.random() < probability
            if should_reply:
                logger.debug(f"概率触发回复，当前概率: {probability}")
            else:
                logger.debug(f"概率回复未触发，当前概率: {probability}")
            return should_reply
        
        # 为未来扩展预留接口
        # 可以在这里添加更多回复方法的判断逻辑
        
        return False

    @staticmethod
    def _check_smart_probability(event: AstrMessageEvent, config: AstrBotConfig, frequency_config: Dict[str, Any]) -> bool:
        """
        使用本地轻量信息动态判断是否回复，避免每条消息都访问大模型。
        """
        if event.is_private_chat():
            logger.debug("私聊消息，智能概率固定回复")
            return True

        smart_config = frequency_config.get("smart_probability", {})
        chat_key = ReplyDecision._get_chat_key(event)
        sender_id = str(event.get_sender_id() or "")
        message_text = event.get_message_outline() or ""

        wakeup_keywords = smart_config.get("wakeup_keywords", [])
        if wakeup_keywords and ReplyDecision._contains_any(message_text, wakeup_keywords):
            ReplyDecision._set_wakeup_state(chat_key, sender_id, message_text, smart_config, event)
            if smart_config.get("reply_on_wakeup", False):
                logger.debug("命中唤醒词，配置为立即回复")
                return True
            logger.debug("命中唤醒词，已进入唤醒判断窗口")
            return False

        if ReplyDecision._get_wakeup_state(chat_key, sender_id):
            logger.debug("处于唤醒窗口内，同一用户后续消息交给大模型判断是否仍在请求范围内")
            return True

        if ReplyDecision._is_at_bot(event):
            probability = ReplyDecision._clamp_float(smart_config.get("direct_mention_probability", 1.0), 0.0, 1.0)
            should_reply = random.random() < probability
            logger.debug(f"消息@机器人，回复概率: {probability}，结果: {should_reply}")
            return should_reply

        if ReplyDecision._is_reply_to_bot(event):
            probability = ReplyDecision._clamp_float(smart_config.get("reply_to_bot_probability", 0.95), 0.0, 1.0)
            should_reply = random.random() < probability
            logger.debug(f"消息引用机器人，回复概率: {probability}，结果: {should_reply}")
            return should_reply

        keywords = frequency_config.get("keywords", [])
        if keywords and ReplyDecision._contains_any(message_text, keywords):
            probability = ReplyDecision._clamp_float(smart_config.get("keyword_probability", 1.0), 0.0, 1.0)
            should_reply = random.random() < probability
            logger.debug(f"命中回复关键词，智能回复概率: {probability}，结果: {should_reply}")
            return should_reply

        probability, reasons = ReplyDecision._calculate_smart_probability(event, smart_config)
        should_reply = random.random() < probability
        logger.debug(f"智能概率回复判断: probability={probability:.3f}, should_reply={should_reply}, reasons={reasons}")
        return should_reply

    @staticmethod
    def _calculate_smart_probability(event: AstrMessageEvent, smart_config: Dict[str, Any]) -> Tuple[float, List[str]]:
        now = time.time()
        chat_key = ReplyDecision._get_chat_key(event)
        sender_id = str(event.get_sender_id() or "")
        text = event.get_message_outline() or ""
        stripped_text = text.strip()
        reasons: List[str] = []

        probability = ReplyDecision._clamp_float(smart_config.get("base_probability", 0.06), 0.0, 1.0)
        reasons.append(f"base={probability:.3f}")

        preferred_keywords = smart_config.get("preferred_keywords", [])
        if preferred_keywords and ReplyDecision._contains_any(text, preferred_keywords):
            bonus = ReplyDecision._to_float(smart_config.get("preferred_topic_bonus", 0.15), 0.15)
            probability += bonus
            reasons.append(f"preferred_topic+{bonus:.3f}")

        weak_keywords = smart_config.get("weak_keywords", [])
        if weak_keywords and ReplyDecision._contains_any(text, weak_keywords):
            penalty = ReplyDecision._to_float(smart_config.get("weak_topic_penalty", 0.25), 0.25)
            probability -= penalty
            reasons.append(f"weak_topic-{penalty:.3f}")

        if ReplyDecision._looks_like_question(stripped_text):
            bonus = ReplyDecision._to_float(smart_config.get("question_bonus", 0.12), 0.12)
            probability += bonus
            reasons.append(f"question+{bonus:.3f}")

        short_threshold = int(smart_config.get("short_message_threshold", 3))
        if 0 < len(stripped_text) <= short_threshold:
            penalty = ReplyDecision._to_float(smart_config.get("short_message_penalty", 0.08), 0.08)
            probability -= penalty
            reasons.append(f"short-{penalty:.3f}")

        activity_window = int(smart_config.get("activity_window_seconds", 120))
        recent_messages = ReplyDecision._get_recent_messages(chat_key, now, activity_window)
        if not recent_messages:
            bonus = ReplyDecision._to_float(smart_config.get("quiet_chat_bonus", 0.08), 0.08)
            probability += bonus
            reasons.append(f"quiet+{bonus:.3f}")
        else:
            last_ts, last_sender_id = recent_messages[-1]
            followup_window = int(smart_config.get("same_user_followup_window_seconds", 120))
            if last_sender_id == sender_id and now - last_ts <= followup_window:
                bonus = ReplyDecision._to_float(smart_config.get("same_user_followup_bonus", 0.08), 0.08)
                probability += bonus
                reasons.append(f"same_user_followup+{bonus:.3f}")

            active_threshold = int(smart_config.get("active_message_threshold", 8))
            active_senders_threshold = int(smart_config.get("active_senders_threshold", 3))
            unique_senders = {sid for _, sid in recent_messages if sid}
            if len(recent_messages) >= active_threshold and len(unique_senders) >= active_senders_threshold:
                penalty = ReplyDecision._to_float(smart_config.get("active_chat_penalty", 0.10), 0.10)
                probability -= penalty
                reasons.append(f"active_chat-{penalty:.3f}")

        reply_window = int(smart_config.get("reply_window_seconds", 300))
        recent_replies = ReplyDecision._get_recent_reply_times(chat_key, now, reply_window)
        bot_context = ReplyDecision._get_recent_bot_context_by_key(
            chat_key,
            now,
            int(smart_config.get("bot_followup_window_seconds", 180))
        )
        if bot_context:
            bonus = ReplyDecision._to_float(smart_config.get("bot_followup_bonus", 0.25), 0.25)
            probability += bonus
            reasons.append(f"bot_followup+{bonus:.3f}")

        if recent_replies:
            cooldown_seconds = int(smart_config.get("cooldown_seconds", 60))
            seconds_since_last_reply = now - recent_replies[-1]
            if seconds_since_last_reply < cooldown_seconds and not bot_context:
                penalty = ReplyDecision._to_float(smart_config.get("cooldown_penalty", 0.25), 0.25)
                probability -= penalty
                reasons.append(f"cooldown-{penalty:.3f}")

            max_replies = int(smart_config.get("max_replies_per_window", 3))
            if max_replies > 0 and len(recent_replies) >= max_replies:
                penalty = ReplyDecision._to_float(smart_config.get("overactive_penalty", 0.20), 0.20)
                probability -= penalty
                reasons.append(f"overactive-{penalty:.3f}")

        user_state = ReplyDecision._user_interactions.get(chat_key, {}).get(sender_id)
        if user_state:
            affinity_window = int(smart_config.get("user_affinity_window_seconds", 1800))
            if now - user_state.get("last_reply_at", 0) <= affinity_window:
                bonus = ReplyDecision._to_float(smart_config.get("user_affinity_bonus", 0.10), 0.10)
                probability += bonus
                reasons.append(f"user_affinity+{bonus:.3f}")

        min_probability = ReplyDecision._clamp_float(smart_config.get("min_probability", 0.0), 0.0, 1.0)
        max_probability = ReplyDecision._clamp_float(smart_config.get("max_probability", 0.45), 0.0, 1.0)
        if max_probability < min_probability:
            max_probability = min_probability

        probability = ReplyDecision._clamp_float(probability, min_probability, max_probability)
        reasons.append(f"clamped={probability:.3f}")
        return probability, reasons
    
    @staticmethod
    def _check_keywords(event: AstrMessageEvent, keywords: list) -> bool:
        """
        检查消息是否包含关键词
        
        Args:
            event: 消息事件
            keywords: 关键词列表
            
        Returns:
            是否包含关键词
        """
        # 获取消息文本
        message_text = event.get_message_outline()
        
        # 检查是否包含关键词
        for keyword in keywords:
            if keyword in message_text:
                return True
                
        return False

    @staticmethod
    def _contains_any(message_text: str, keywords: list) -> bool:
        normalized_text = (message_text or "").lower()
        for keyword in keywords:
            normalized_keyword = str(keyword).strip().lower()
            if normalized_keyword and normalized_keyword in normalized_text:
                return True
        return False
        
    @staticmethod
    def _check_blacklist_keywords(event: AstrMessageEvent, blacklist_keywords: list) -> bool:
        """
        检查消息是否包含黑名单关键词
        
        Args:
            event: 消息事件
            blacklist_keywords: 黑名单关键词列表
            
        Returns:
            是否包含黑名单关键词
        """
        # 获取消息文本
        message_text = event.get_message_outline()
        
        # 检查是否包含黑名单关键词
        for keyword in blacklist_keywords:
            if keyword in message_text:
                return True
                
        return False

    @staticmethod
    def _get_chat_key(event: AstrMessageEvent) -> str:
        platform_name = event.get_platform_name()
        is_private_chat = event.is_private_chat()
        chat_id = event.get_sender_id() if is_private_chat else event.get_group_id()
        return LLMUtils.get_chat_key(platform_name, is_private_chat, str(chat_id))

    @staticmethod
    def _set_wakeup_state(chat_key: str, sender_id: str, message_text: str, smart_config: Dict[str, Any], event: AstrMessageEvent) -> None:
        message_obj = getattr(event, "message_obj", None)
        ReplyDecision._wakeup_states[chat_key] = {
            "sender_id": sender_id,
            "expires_at": time.time() + int(smart_config.get("wakeup_window_seconds", 90)),
            "message_text": message_text,
            "message_id": getattr(message_obj, "message_id", None),
            "created_at": time.time(),
        }

    @staticmethod
    def _get_wakeup_state(chat_key: str, sender_id: str, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
        if now is None:
            now = time.time()

        state = ReplyDecision._wakeup_states.get(chat_key)
        if not state:
            return None

        if state.get("expires_at", 0) <= now:
            ReplyDecision._wakeup_states.pop(chat_key, None)
            return None

        if state.get("sender_id") != sender_id:
            return None

        return state

    @staticmethod
    def get_wakeup_context(event: AstrMessageEvent) -> Optional[Dict[str, Any]]:
        if event.is_private_chat():
            return None
        chat_key = ReplyDecision._get_chat_key(event)
        sender_id = str(event.get_sender_id() or "")
        state = ReplyDecision._get_wakeup_state(chat_key, sender_id)
        if not state:
            return None

        message_obj = getattr(event, "message_obj", None)
        current_message_id = getattr(message_obj, "message_id", None)
        is_trigger_message = bool(current_message_id and current_message_id == state.get("message_id"))
        if not is_trigger_message and not current_message_id:
            current_text = event.get_message_outline() or ""
            is_trigger_message = current_text == state.get("message_text") and time.time() - state.get("created_at", 0) < 2
        state["is_trigger_message"] = is_trigger_message
        return state

    @staticmethod
    def get_recent_bot_context(event: AstrMessageEvent, window_seconds: int = 180) -> Optional[Dict[str, Any]]:
        if event.is_private_chat():
            return None
        chat_key = ReplyDecision._get_chat_key(event)
        return ReplyDecision._get_recent_bot_context_by_key(chat_key, time.time(), window_seconds)

    @staticmethod
    def _is_at_bot(event: AstrMessageEvent) -> bool:
        message_obj = getattr(event, "message_obj", None)
        message_chain = getattr(message_obj, "message", []) if message_obj else []
        self_id = str(event.get_self_id() or "")
        if not self_id:
            return False

        for component in message_chain:
            component_type = getattr(component, "type", None) or component.__class__.__name__.lower()
            if component_type != "at" and not isinstance(component, At):
                continue
            target_id = str(getattr(component, "qq", "") or getattr(component, "id", ""))
            if target_id == self_id:
                return True
        return False

    @staticmethod
    def _is_reply_to_bot(event: AstrMessageEvent) -> bool:
        message_obj = getattr(event, "message_obj", None)
        message_chain = getattr(message_obj, "message", []) if message_obj else []
        self_id = str(event.get_self_id() or "")
        if not self_id:
            return False

        for component in message_chain:
            component_type = getattr(component, "type", None) or component.__class__.__name__.lower()
            if component_type != "reply" and not isinstance(component, Reply):
                continue
            sender_id = str(getattr(component, "sender_id", "") or "")
            if sender_id == self_id:
                return True
        return False

    @staticmethod
    def _looks_like_question(message_text: str) -> bool:
        if not message_text:
            return False
        question_marks = ("?", "？", "吗", "么", "啥", "什么", "怎么", "如何", "为什么", "哪", "谁")
        return any(mark in message_text for mark in question_marks)

    @staticmethod
    def _remember_user_message(event: AstrMessageEvent) -> None:
        if event.is_private_chat():
            return

        now = time.time()
        chat_key = ReplyDecision._get_chat_key(event)
        sender_id = str(event.get_sender_id() or "")
        records = ReplyDecision._recent_messages.setdefault(chat_key, [])
        records.append((now, sender_id))
        ReplyDecision._recent_messages[chat_key] = records[-80:]

    @staticmethod
    def _get_recent_messages(chat_key: str, now: float, window_seconds: int) -> List[Tuple[float, str]]:
        records = ReplyDecision._recent_messages.get(chat_key, [])
        recent = [(ts, sender_id) for ts, sender_id in records if now - ts <= window_seconds]
        ReplyDecision._recent_messages[chat_key] = recent
        return recent

    @staticmethod
    def _get_recent_reply_times(chat_key: str, now: float, window_seconds: int) -> List[float]:
        records = ReplyDecision._recent_reply_times.get(chat_key, [])
        recent = [ts for ts in records if now - ts <= window_seconds]
        ReplyDecision._recent_reply_times[chat_key] = recent
        return recent

    @staticmethod
    def _record_bot_reply(event: AstrMessageEvent) -> None:
        if event.is_private_chat():
            return

        now = time.time()
        chat_key = ReplyDecision._get_chat_key(event)
        sender_id = str(event.get_sender_id() or "")

        reply_times = ReplyDecision._recent_reply_times.setdefault(chat_key, [])
        reply_times.append(now)
        ReplyDecision._recent_reply_times[chat_key] = reply_times[-30:]

        chat_interactions = ReplyDecision._user_interactions.setdefault(chat_key, {})
        user_state = chat_interactions.setdefault(sender_id, {"reply_count": 0, "last_reply_at": 0})
        user_state["reply_count"] = int(user_state.get("reply_count", 0)) + 1
        user_state["last_reply_at"] = now

    @staticmethod
    def record_bot_message_sent(event: AstrMessageEvent, chain: List[BaseMessageComponent]) -> None:
        if event.is_private_chat():
            return

        ReplyDecision._record_bot_reply(event)
        chat_key = ReplyDecision._get_chat_key(event)
        message_text = ReplyDecision._outline_text_chain(chain)
        if len(message_text) > 300:
            message_text = message_text[:300] + "..."

        ReplyDecision._last_bot_messages[chat_key] = {
            "timestamp": time.time(),
            "message_text": message_text,
            "trigger_sender_id": str(event.get_sender_id() or ""),
        }

    @staticmethod
    def _get_recent_bot_context_by_key(chat_key: str, now: float, window_seconds: int) -> Optional[Dict[str, Any]]:
        context = ReplyDecision._last_bot_messages.get(chat_key)
        if not context:
            return None
        if now - context.get("timestamp", 0) > window_seconds:
            ReplyDecision._last_bot_messages.pop(chat_key, None)
            return None
        return context

    @staticmethod
    def _outline_text_chain(chain: List[BaseMessageComponent]) -> str:
        parts = []
        for component in chain or []:
            text = getattr(component, "text", None)
            if text:
                parts.append(str(text))
        return "".join(parts).strip()

    @staticmethod
    def _to_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clamp_float(value: Any, minimum: float, maximum: float) -> float:
        numeric_value = ReplyDecision._to_float(value, minimum)
        return max(minimum, min(maximum, numeric_value))

    @staticmethod
    async def process_and_reply(event: AstrMessageEvent, config: AstrBotConfig, context: Context):
        """
        处理消息并使用大模型回复
        
        Args:
            event: 消息事件
            config: 配置对象
            context: 上下文对象
            
        Yields:
            大模型的回复
        """
        # 获取必要信息
        platform_name = event.get_platform_name()
        is_private = event.is_private_chat()
        chat_id = event.get_sender_id() if is_private else event.get_group_id()

        # 标记开始处理
        LLMUtils.set_llm_in_progress(platform_name, is_private, chat_id)

        try:
            # 调用大模型并发送回复
            request = await LLMUtils.call_llm(event, config, context)
            yield request
        finally:
            # 标记处理完成
            LLMUtils.set_llm_in_progress(platform_name, is_private, chat_id, False)
