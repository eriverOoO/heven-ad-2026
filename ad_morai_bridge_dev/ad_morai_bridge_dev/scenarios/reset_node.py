from pathlib import Path

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from ad_morai_bridge_dev.simulator_grpc.client import MoraiGrpcClient
from ad_morai_bridge_dev.simulator_grpc.descriptors import MoraiApi
from ad_morai_bridge_dev.scenarios.reset import load_reset_plan, reset_scenario


class ScenarioResetNode(Node):
    def __init__(self):
        super().__init__("ad_morai_scenario_reset")
        scenario_file = str(
            self.declare_parameter("scenario_file", "").value
        )
        target = str(
            self.declare_parameter("grpc.target", "127.0.0.1:7789").value
        )
        self._timeout = float(
            self.declare_parameter("grpc.timeout_sec", 5.0).value
        )
        if not scenario_file:
            raise ValueError("scenario_file is empty")

        self._plan = load_reset_plan(Path(scenario_file))
        self._client = MoraiGrpcClient.connect(
            MoraiApi.load(), target, default_timeout=self._timeout
        )
        self._service = self.create_service(
            Trigger, "/ad/dev/scenario/reset", self._on_reset
        )

    def _on_reset(self, _request, response):
        try:
            reset_scenario(self._client, self._plan, self._timeout)
            response.success = True
            response.message = "MORAI scenario actors reset"
        except Exception as exc:
            response.success = False
            response.message = str(exc)
        return response

    def destroy_node(self):
        self._client.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ScenarioResetNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
