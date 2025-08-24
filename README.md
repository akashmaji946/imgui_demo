## Configure

```bash
cmake -S src -B build

[or]

./c
```

## Build

```bash
cmake --build build --target cuda_demo -j8

[or]

./b
```

## Run

```bash
./build/cuda_demo models/teapot.off

[or]

./r models/teapot.off
```

## Clean
```bash
rm -rf build

[or]

./c
```
