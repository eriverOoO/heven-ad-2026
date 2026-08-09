import rclpy
from rclpy.node import Node

from ad_morai_bridge_dev.simulator_grpc.client import MoraiGrpcClient
from ad_morai_bridge_dev.simulator_grpc.descriptors import MoraiApi
from ad_morai_bridge_dev.scenarios.setup import load_scenario_and_resume


class ScenarioSetupNode(Node):
    def __init__(self):
        super().__init__("ad_morai_scenario_setup")
        self._scenario_file = str(
            self.declare_parameter("scenario_file", "").value
        )
        self._grpc_target = str(
            self.declare_parameter("grpc.target", "127.0.0.1:7789").value
        )
        self._timeout_sec = float(
            self.declare_parameter("grpc.timeout_sec", 8.0).value
        )

    def execute(self) -> None:
        if not self._scenario_file:
            raise RuntimeError("scenario_file is empty")
        client = MoraiGrpcClient.connect(
            MoraiApi.load(),
            self._grpc_target,
            default_timeout=self._timeout_sec,
        )
        try:
            load_scenario_and_resume(
                client, self._scenario_file, self._timeout_sec
            )
        finally:
            client.close()
        self.get_logger().info(
            f"loaded and resumed MORAI scenario: {self._scenario_file}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = None
    exit_code = 1
    try:
        node = ScenarioSetupNode()
        node.execute()
        exit_code = 0
    except Exception as exc:
        if node is not None:
            node.get_logger().error(f"scenario setup failed: {exc}")
        else:
            rclpy.logging.get_logger("ad_morai_scenario_setup").error(
                "scenario setup failed before node initialization: "
                f"{exc}"
            )
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
