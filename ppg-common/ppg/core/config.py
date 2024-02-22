import enum

class OriginationEnum(str, enum.Enum):
    ORIGINATED_FROM_APP = "app"
    ORIGINATED_FROM_TEST = "test"
