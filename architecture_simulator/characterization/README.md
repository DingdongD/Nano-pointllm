# Characterization Inputs

Area and energy outputs are disabled by default. Enable them only with a record
containing process node, tool/version, corner, voltage, frequency, activity
source, SRAM compiler or CACTI configuration, and per-block provenance.

`HardwareConfig.characterization` accepts:

- `mac_energy_pj`: precision-indexed energy per MAC;
- `hbm_energy_pj_per_byte`;
- `sram_energy_pj_per_byte`;
- `block_area_um2`: stable RTL block-name to area mapping.

The intended flow mirrors the local PointAcc projects: RTL simulation produces
named activity, PTPX maps activity to block power, and this simulator imports
only reviewed aggregate coefficients. Missing fields remain `null`; they are
never silently replaced with guessed values.
