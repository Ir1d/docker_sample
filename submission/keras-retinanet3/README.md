# Keras-retinanet

Docker sample of keras-retinanet for UG^{2+} Track 2

https://github.com/Ir1d/docker_sample/tree/master/submission/keras-retinanet

https://hub.docker.com/r/scaffrey/keras-retinanet_sample

## Usage

**submission images are required to read input from `/images` and save results to `/predictions`**

### SUB-CHALLENGE 2.1

```python
python demo1.py
```

```bash
time docker run --rm -it scaffrey/keras-retinanet_sample demo1.py
```

The prediction goes to `/predictions` folder, you can compare them to `/predictions-ref` folder.

### SUB-CHALLENGE 2.3

**please note that submissions in SUB-CHALLENGE 2.3 are requireed to read from `/images3` and save to `/predictions3`**

```python
python demo3.py
```

```bash
time docker run --rm -it scaffrey/keras-retinanet_sample demo3.py
```

The prediction goes to `/predictions3` folder, you can compare them to `/predictions3-ref` folder.
