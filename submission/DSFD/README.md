# DSFD

Docker sample of DSFD for UG^{2+} Track 2

https://github.com/Ir1d/docker_sample/tree/master/submission/DSFD

https://hub.docker.com/r/scaffrey/dsfd_sample

## Usage

**submission images are required to have a /run.sh and read input from the first argument and save results to the second one**

```bash
docker pull scaffrey/dsfd_sample:leinao
nvidia-docker run -ti -v $user_output_path2:/predictions/ -v $user_input_path2:/images/ --entrypoint /bin/bash scaffrey/dsfd_sample:leinao /run.sh /images /predictions
```
