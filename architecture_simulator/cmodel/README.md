# Event-Level C-model Contract

This directory is reserved for the C++ or SystemC timing model that replaces
individual Python engines without changing workload/config/result JSON.

The structure follows the useful separation in `/home/Compiler_Codes`: typed
operations, explicit memory requests, independent engines, and a scheduler that
owns timestamps. It must not import RTL internals into the workload format.

Required engine events:

- `issue(op_id, phase, shape, precision, mapping)`
- `memory_request(op_id, space, address, bytes, read_write)`
- `engine_start(op_id, engine, cycle)`
- `engine_complete(op_id, engine, cycle)`
- `counter(op_id, name, value)`

Promotion gate: for each engine, the C-model must match the checked Python
traffic counts exactly and match an RTL trace within the tolerance documented
in `../verification/README.md`.
