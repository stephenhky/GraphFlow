
class UnknownNodeException(Exception):
    def __init__(self, nodes: list[str]):
        self.message = f"Unknown nodes: {', '.join(nodes)}"
