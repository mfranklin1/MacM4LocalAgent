# MacM4LocalAgent — Documentation Index

Welcome. This folder is the deep dive. Skim the [project README](../README.md)
first for the elevator pitch, then come back here for the details.

## Reading order

> **First-time install?** Read [Cursor setup runbook](RUNBOOK-cursor-setup.md)
> — the step-by-step walkthrough for wiring Cursor to the local proxy with
> the values from your install.

1. [Architecture](architecture.md) — the big picture, components and dataflow.
2. [Routing](routing.md) — how `hybrid-auto` decides between local-fast,
   local-long and Claude.
3. [Offline mode](offline-mode.md) — airplane / no-network behavior,
   `make offline`, and the recipe for clearing Cline context cleanly
   when the network drops mid-task.
3. [Cost model](cost-model.md) — how `actual_cost`, `shadow_cost` and savings
   are computed and stored.
4. [Operations](operations.md) — `make` targets, `launchd` services and logs.
5. [Cursor integration](cursor-integration.md) — wiring Cursor IDE to the local
   LiteLLM proxy.
6. [Testing](testing.md) — how the test suite is organized and what each
   suite covers.
7. [Troubleshooting](troubleshooting.md) — fix the most common breakages.
8. [FAQ](faq.md) — short answers to the questions that come up most often.
9. [Contributing](contributing.md) — how to make changes safely.
10. [Security](security.md) — what stays on-device, what leaves it.

## At-a-glance file map

| Topic                | File                              |
| -------------------- | --------------------------------- |
| Cursor setup runbook | [RUNBOOK-cursor-setup.md](RUNBOOK-cursor-setup.md) |
| Architecture diagram | [architecture.md](architecture.md) |
| Routing decisions    | [routing.md](routing.md)          |
| Offline / airplane   | [offline-mode.md](offline-mode.md) |
| Cost / savings math  | [cost-model.md](cost-model.md)    |
| Service management   | [operations.md](operations.md)    |
| Cursor IDE setup     | [cursor-integration.md](cursor-integration.md) |
| Test layout          | [testing.md](testing.md)          |
| Common breakages     | [troubleshooting.md](troubleshooting.md) |
| FAQ                  | [faq.md](faq.md)                  |
| Contributing guide   | [contributing.md](contributing.md) |
| Security model       | [security.md](security.md)        |
| Changelog            | [../CHANGELOG.md](../CHANGELOG.md) |
