#!/usr/bin/env bash

# Check args
if [ "$#" -ne 1 ]; then
	  echo "usage: ./run.sh IMAGE_NAME"
	    return 1
    fi

    # Get this script's path
    pushd `dirname $0` > /dev/null
    SCRIPTPATH=`pwd`
    popd > /dev/null
    
    set -e

    # Run the container with shared X11
    docker run\
	        --gpus all\
		  --publish-all=true\
		    --net=host\
			--mount type=bind,source=/path/to/repository,target=/repositories/TODO \
			  	  -it $1
    # -v /tmp/.X11-unix:/tmp/.X11-unix \
  				# -e DISPLAY=$DISPLAY \
  				# -e QT_X11_NO_MITSHM=1 \
				# -e SHELL\
			    # -e DOCKER=1\
			    
		        
		      
