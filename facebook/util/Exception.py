from enum import Enum


class TException(Enum):

    # 没有更多的推文可以被获取
    NO_NEW_CONTENT = -2

    # 标签结构与标准推文不一致
    STRUCTURE_EXCEPTION = -1
