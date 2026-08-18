"""LLM-generated customer replies."""

from collections.abc import Iterator

from llm_client import LLMClient


SYSTEM_PROMPT = """你是一名友好、专业的产品顾问。
请先直接回答客户的问题，再自然地引导下一步。只使用提供的产品资料，不要编造；
资料中没有答案时应坦诚说明。回复使用客户的语言，保持简洁，不要使用 Markdown。"""


PRODUCT_DESCRIPTION = """产品名称：MoriAir P1 智能桌面植物舱。

MoriAir P1 是一款用于家庭和小型办公室的水培种植设备，适合没有园艺经验、
但希望全年种植新鲜香草和叶菜的用户。机身尺寸为 42 × 21 × 38 厘米，空机重
4.2 千克，配有 12 升水箱和 8 个独立种植位，可种植罗勒、薄荷、生菜、芝麻菜、
香菜和小番茄等作物；不适合土豆、胡萝卜等大型根茎作物。

设备使用 24W 全光谱 LED 灯，默认每天自动照明 16 小时，灯架最高可升至 32 厘米。
循环水泵每 30 分钟运行 5 分钟，工作噪声低于 28 分贝。水位过低时，机身指示灯和
手机应用都会提醒用户。设备支持 2.4GHz Wi-Fi，可在应用中调整照明时间、查看水位、
记录播种日期和接收营养液添加提醒；核心种植功能不依赖应用，也不需要订阅服务。

标准售价为 899 元，包装内含植物舱、8 个种植篮、24 个育苗海绵、A/B 营养液各一瓶、
罗勒和生菜种子各一包以及电源适配器。正常使用时平均耗电约 0.35 度/天。水箱通常每
2 至 3 周清洗一次，水泵滤网可拆卸冲洗，其他耗材可以单独购买，也兼容常见水培耗材。

中国大陆地区包邮，通常付款后 48 小时内发货。未使用且包装完整的产品支持 7 天无理由
退货；整机保修 1 年，LED 灯板和水泵保修 2 年。产品使用 24V 低压电源，但机身不防水，
清洁前必须断电，不能放在露天、浴室或儿童容易碰倒的位置。"""


class ReplyService:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def reply(self, user_message: str) -> str:
        return "".join(self.stream(user_message))

    def stream(self, user_message: str) -> Iterator[str]:
        yield from self.llm.stream(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "system", "content": PRODUCT_DESCRIPTION},
                {"role": "user", "content": user_message},
            ]
        )
