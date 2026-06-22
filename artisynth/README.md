# ArtiSynth custom sources

Drop these into your ArtiSynth tree (preserving package paths) and recompile:

    artisynth_core/src/artisynth/models/tongue3d/FemTongueMriDemo.java
    artisynth_core/src/artisynth/models/jawTongue/JawFemMuscleTongueMriDemo.java
    artisynth_core/src/artisynth/models/jawTongue/MriSequence{Manifest,IO,Data}.java
    artisynth_core/src/artisynth/models/jawTongue/MriRegistration2d.java

Then compile (Linux: `make`; Windows: `bin\compile.bat`) and load
`artisynth.models.tongue3d.FemTongueMriDemo` (tongue-only, metres) or
`artisynth.models.jawTongue.JawFemMuscleTongueMriDemo` (jaw+tongue, mm).
