from dataclasses import dataclass

from google.protobuf import json_format, message_factory

from ad_morai_bridge_dev.simulator_grpc.descriptors import MoraiApi


@dataclass(frozen=True)
class GrpcCallResult:
    success: bool
    code: int
    status: str
    response_json: str = ""
    error: str = ""


class MoraiGrpcClient:
    def __init__(
        self,
        api: MoraiApi,
        channel,
        *,
        default_timeout: float = 5.0,
        grpc_module=None,
    ):
        if grpc_module is None:
            import grpc as grpc_module

        self._api = api
        self._channel = channel
        self._default_timeout = default_timeout
        self._grpc = grpc_module
        self._factory = message_factory.MessageFactory(api.pool)

    @classmethod
    def connect(
        cls, api: MoraiApi, target: str, *, default_timeout: float = 5.0
    ) -> "MoraiGrpcClient":
        import grpc

        return cls(
            api,
            grpc.insecure_channel(target),
            default_timeout=default_timeout,
            grpc_module=grpc,
        )

    def call_json(
        self,
        method: str,
        request_json: str,
        timeout: float | None = None,
    ) -> GrpcCallResult:
        try:
            descriptor = self._api.resolve_method(method)
            request_type = self._message_type(descriptor.input_type)
            response_type = self._message_type(descriptor.output_type)
            request = request_type()
            json_format.Parse(
                request_json,
                request,
                descriptor_pool=self._api.pool,
            )
        except (ValueError, json_format.ParseError) as exc:
            return GrpcCallResult(False, 3, "INVALID_ARGUMENT", error=str(exc))

        canonical_path = f"/{descriptor.containing_service.full_name}/{descriptor.name}"
        call = self._channel.unary_unary(
            canonical_path,
            request_serializer=lambda message: message.SerializeToString(),
            response_deserializer=response_type.FromString,
        )
        call_timeout = timeout if timeout is not None and timeout > 0.0 else self._default_timeout
        try:
            response = call(request, timeout=call_timeout)
        except self._grpc.RpcError as exc:
            code = exc.code()
            value = code.value[0] if isinstance(code.value, tuple) else int(code.value)
            return GrpcCallResult(
                False,
                value,
                code.name,
                error=exc.details() or "",
            )

        return GrpcCallResult(
            True,
            0,
            "OK",
            response_json=json_format.MessageToJson(
                response,
                including_default_value_fields=True,
                preserving_proto_field_name=True,
                indent=None,
                descriptor_pool=self._api.pool,
            ),
        )

    def close(self) -> None:
        self._channel.close()

    def _message_type(self, descriptor):
        if hasattr(message_factory, "GetMessageClass"):
            return message_factory.GetMessageClass(descriptor)
        return self._factory.GetPrototype(descriptor)
