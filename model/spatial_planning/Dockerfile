FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

# Note: we remove the files added by "apt-get update" in the same build command
# to avoid adding them to this layer (each command is a layer). If a file is
# added and removed in the same build command, they won't appear in the "diff"
# of the layer. This is optional but will keep the image as small as possible.
RUN apt-get update && apt-get install -y --no-install-recommends \
      wget \
      python3.7 \
      python3-pip \
      python3-dev \
      python2.7 \
      libsm6 \
      libxext6 \
      vim \
      libxrender-dev \
      libgl1-mesa-glx \
      python3-setuptools && \
      rm -rf /var/lib/apt/lists/*

RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.6/julia-1.6.1-linux-x86_64.tar.gz && \
      tar zxvf julia-1.6.1-linux-x86_64.tar.gz


# Copy your source code using paths relative to your project root. The second
# argument is the destination path within the container.
COPY . /spatial-planning/


ENV PYTHONPATH="/spatial-planning/:${PYTHONPATH}"
ENV PATH="${PATH}:/julia-1.6.1/bin"

RUN python3 -m pip install -r spatial-planning/requirements.txt
RUN julia /spatial-planning/packages.jl


ENTRYPOINT cd spatial-planning && python-jl notebooks/similarity_plots.py --task-name=$task_name --exp-name=$exp_name --agent-name=$agent_name --search-depth=$search_depth --tree-queries=$tree_queries --discount-factor=$discount_factor --observation_mode=$observation_mode --optimism=$optimism --num_iterations=$num_iterations --participant=$participant --save-dir=$save_dir --replan-strat=$replan_strat
# ENTRYPOINT ["tail", "-f", "/dev/null"]