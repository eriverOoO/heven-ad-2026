# CI dependency image

Pull request CI uses a private, source-free ROS 2 Humble dependency image from
GitHub Container Registry (GHCR). The repository source and `build/`, `install/`
outputs are not stored in the image. Every pull request still checks out its exact
commit and runs the complete build and test suite.

## Reuse rule

`scripts/ci_image_fingerprint.py` hashes the files that can change the installed
dependency set:

- `docker/ci/Dockerfile` and `docker/ci/rebuild-revision`
- `dependencies.repos`
- every tracked ROS `package.xml`
- the dependency patch script and patch files

Application source changes do not change this key. CI reuses
`ghcr.io/skku-heven/heven-ad-2026-ci:humble-<fingerprint>` when it already exists.
The build job consumes the resolved image digest, rather than a mutable tag.

Changing a `package.xml`, pinned external repository, patch, or the Dockerfile
automatically creates a new dependency image. If only the Ubuntu/ROS APT index has
changed, run the CI workflow manually with `rebuild_image=true`, or increment
`docker/ci/rebuild-revision`. The weekly scheduled run also refreshes APT packages
without running the source build job.

## Local verification

```bash
cd /home/heven/heven_ad_2026_ws/src/heven_ad_2026
fingerprint="$(python3 scripts/ci_image_fingerprint.py --root .)"
docker buildx build \
  --file docker/ci/Dockerfile \
  --build-arg "APT_REFRESH=$(date -u +%Y%m%d)" \
  --build-arg "DEPENDENCY_FINGERPRINT=$fingerprint" \
  --tag "heven-ad-2026-ci:humble-$fingerprint" \
  --load \
  .
```

The GHCR package should remain private and inherit access from this repository.
No personal access token or repository source is embedded in the image.

## GitHub-hosted runner measurements

Measurements below used the same PR head (`c4c8ebf`) on GitHub's
`ubuntu-22.04` hosted runner. Both attempts ran the complete HEVEN build closure
and all 2,215 tests. The dependency image already existed in both attempts, so
the comparison isolates compiler-cache reuse.

Run: [31081629363](https://github.com/skku-heven/heven-ad-2026/actions/runs/31081629363)

| Condition | Image lookup | Cache restore | Build | Test | Build job | Workflow |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| warm image, cold ccache (attempt 1) | 14s | 1s | 51m 07s | 1m 31s | 54m 41s | 55m 01s |
| warm image, warm ccache (attempt 2) | 17s | 10s | 3m 38s | 1m 30s | 6m 53s | 7m 16s |

The warm run hit all 695 compiler requests. Compared with the cold-ccache
attempt, compiler-cache reuse reduced the build step by 92.9% (14.1x) and the
complete workflow by 86.9% (7.6x). Tests remain uncached and therefore retain
their full validation cost.

Creating the dependency image itself is a separate cold cost. The first image
build in [run 31077662275](https://github.com/skku-heven/heven-ad-2026/actions/runs/31077662275)
took 9m 13s. Subsequent PR runs resolved the existing immutable image in
14-17 seconds.

Hosted-runner performance varies, so these values are evidence from the linked
runs rather than a duration guarantee. Dependency changes create a new image;
compiler-affecting changes naturally reduce the ccache hit rate.
