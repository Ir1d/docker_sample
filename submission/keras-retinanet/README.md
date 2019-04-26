# Keras-retinanet

Docker sample of keras-retinanet for UG^{2+} Track 2

https://github.com/Ir1d/docker_sample/tree/master/submission/keras-retinanet

https://hub.docker.com/r/scaffrey/keras-retinanet_sample

## Usage

**submission images are required to have a /run.sh and read input from the first argument and save results to the second one**

```bash
docker pull scaffrey/keras-retinanet_sample:leinao
nvidia-docker run -ti -v $user_output_path1:/predictions/ -v $user_input_path1:/images/ --entrypoint /bin/bash scaffrey/keras-retinanet_sample:leinao /run.sh /images /predictions
```
