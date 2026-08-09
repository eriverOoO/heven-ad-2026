# Development-only MORAI actor control

Inspection date: 2026-08-04 KST

This runbook covers only `ad_morai_bridge_dev`. The actor CLI is not imported,
launched, or consumed by the competition bridge or normal competition bringup.
It does not load a map or scenario and it does not publish simulator truth into
localization, perception, planning, or control.

## Safety contract

All network traffic goes through the existing descriptor-driven
`MoraiGrpcClient.call_json(method, request_json, timeout)`. Every call has a
positive finite deadline. The typed wrapper exposes only these exact descriptor
methods:

- `morai_sim_api.simualtor.Simulator/GetAvailableObject`
- `morai_sim_api.actor.Actor/GetAllActorsState`
- `morai_sim_api.actor.Actor/GetActorState`
- `morai_sim_api.actor.Actor/SpawnVehicle`
- `morai_sim_api.actor.Actor/DestroyActor`
- `morai_sim_api.actor.Actor/SetPause`
- `morai_sim_api.actor.Actor/SetTransform`
- `morai_sim_api.actor.Actor/SetVelocity`
- `morai_sim_api.actor.Actor/SetVehicleRoute`

There is no public raw method/path argument. Scenario/map loading, global
`Simulation/Start`, `Stop`, `Pause`, and `Resume`, and
`Actor/DestroyAllActors` are rejected before the transport is called. Spawn and
destroy are non-idempotent and are never automatically retried. A deadline on
either operation is an ambiguous outcome that an operator must reconcile with
read-only `list`/`state` inspection.

`client_key` is an ownership/group label, not authentication. Mutating an NPC
requires a nonempty configured key and a full identity match of actor ID,
object type, and key. A read-only `state`/vehicle lookup never grants mutation
authority. Ego mutation is separate: `--ego-id`, vehicle type, and
`--ego-client-key` form one configured identity, and `hold-ego` or
`teleport-ego` proceeds only after the actor list contains that exact identity.

MORAI returns the runtime spawn ID in `Result.custom_message`; that ID, rather
than the requested ID, is captured immediately as a version 2 spawn manifest.
The manifest is `pending` until an exact `GetActorState` verification succeeds,
then becomes `verified`. A verification failure is nonzero, is never retried,
and emits the pending runtime identity plus `reconciliation_required: true` on
stderr. If `--manifest-out` was supplied, its parent and a temporary file are
preflighted before `SpawnVehicle`; the final path is created atomically without
overwriting an existing file. A post-spawn write failure is also nonzero and
emits the verified runtime identity on stderr for manual recovery.

`destroy-created` accepts either reconciliation state, re-queries the actor,
calls `DestroyActor` for that single exact identity, and verifies its absence.
A same-ID identity mismatch is skipped with an error. A missing actor is
already clean.

## Descriptor provenance

The committed `ad_morai_bridge_dev/data/morai_api.desc` has SHA-256
`764afbae20d60ec4614d6b9abe47218c72925671ffa20773a60f192277b9ce5a`.
It contains 50 protobuf files, 9 services, 93 unary RPCs, and 38 unary Actor
RPCs. `scripts/generate_morai_descriptors.py` extracted the embedded
`FileDescriptorProto` values from the repository-pinned MORAI C# source at
commit `6eafbeb662ce2ed8928cae21b413c0553aaf9cee`; `dependencies.repos` records
that revision. No external repository was cloned or committed for this work.

The installed simulator audited for this contract was
`Simulator_v.S4.251001.MolitComp03_Linux` (25.S4). The newer official Python
branch is a 26.R1 interoperability reference, not a descriptor repin: its Actor
messages matched the committed local Actor descriptors, but other services and
the source layout differ.

## Official primary references

Only official MORAI sources were used to interpret the API. These URLs were
inspected on 2026-08-04 KST:

- Repository and pinned 24.R1 C# commit:
  <https://github.com/MORAI-Autonomous/MORAI-DriveExample_GRPC/commit/6eafbeb662ce2ed8928cae21b413c0553aaf9cee>
- Official 26.R1 Python commit used for current wrapper semantics:
  <https://github.com/MORAI-Autonomous/MORAI-DriveExample_GRPC/commit/1181793de1a5c6ce0c45d84d00f200511012bdd7>
- Actor service/stub at that exact commit:
  <https://github.com/MORAI-Autonomous/MORAI-DriveExample_GRPC/blob/1181793de1a5c6ce0c45d84d00f200511012bdd7/scripts/lib/grpc/src/proto/morai/actor/actor_pb2_grpc.py>
- Official Actor wrapper:
  <https://github.com/MORAI-Autonomous/MORAI-DriveExample_GRPC/blob/1181793de1a5c6ce0c45d84d00f200511012bdd7/scripts/lib/grpc/src/api/actor.py>
- Official vehicle route wrapper:
  <https://github.com/MORAI-Autonomous/MORAI-DriveExample_GRPC/blob/1181793de1a5c6ce0c45d84d00f200511012bdd7/scripts/lib/grpc/src/api/vehicle.py>
- Official lifecycle wrapper whose broad cleanup/start-stop defaults are
  deliberately not copied:
  <https://github.com/MORAI-Autonomous/MORAI-DriveExample_GRPC/blob/1181793de1a5c6ce0c45d84d00f200511012bdd7/scripts/lib/grpc/src/api/simulation_world.py>
- Official Standard gRPC overview:
  <https://help-morai-sim-en.scrollhelp.site/morai-sim-standard-en/grpc-api>

At inspection, the official repository had no `LICENSE` file and GitHub showed
no detected license. Therefore this Apache-2.0 package contains an independent
wrapper and no copied upstream wrapper implementation. The already committed
generated descriptor remains a derived upstream artifact; maintainers must
resolve upstream redistribution terms before distributing it beyond the
existing project boundary.

## Actor-preset map provenance

`config/actor_presets.provenance.yaml` records the source map identity, the
SHA-256 of the curated `ad_data/map/link_set.json` snapshot, and the exact
ordered `(link ID, waypoint index)` occurrences, node endpoints, selected
waypoint coordinates, and forward heading for every route entry. Repeated link
occurrences remain ordered and are validated independently. The source map
itself is committed through Git LFS. The repository test verifies its hash,
waypoint range, and selected-link geometry together with the preset evidence:

```bash
./scripts/test_python.sh \
  ad_morai_bridge_dev/test/test_actor_provenance.py
```

## Commands

Inspect the exact JSON plan without loading descriptors, opening a channel, or
calling the simulator:

```bash
ad_morai_actor --dry-run models
ad_morai_actor --dry-run list
ad_morai_actor --dry-run spawn-npc \
  --preset kcity-roundabout-loop \
  --manifest-out /tmp/heven-roundabout-created.json
ad_morai_actor --dry-run \
  --ego-id Ego --ego-client-key '' hold-ego
```

The exact supported commands are `models`, `list`, `hold-ego`,
`teleport-ego`, `spawn-npc`, `route-npc`, `pause-actor`, `resume-actor`,
`state`, and `destroy-created`. Omitting `--dry-run` performs the requested
calls, so first inspect the plan and current actors. A safe controlled lifecycle
is:

```bash
ad_morai_actor list
ad_morai_actor spawn-npc \
  --preset kcity-roundabout-loop \
  --manifest-out /tmp/heven-roundabout-created.json
ad_morai_actor route-npc \
  --manifest /tmp/heven-roundabout-created.json \
  --preset kcity-roundabout-loop
ad_morai_actor destroy-created \
  --manifest /tmp/heven-roundabout-created.json
```

The preset file contains a closed K-City roundabout link cycle and a
map-derived highway route whose curated-map evidence is pinned by the
separate provenance manifest. It never requests a map load. Before execution,
compare `models` with the preset model and confirm the active map is
`R_KR_PR_K-city_2025`. `SetVelocity` is an initial/current actor value, not
proof that physical AI will maintain an exact target speed; measure the
returned `global_velocity` when evaluating motion.

No live simulator mutation was performed while implementing this toolkit.
