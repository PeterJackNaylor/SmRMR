FROM ubuntu:latest

# install anaconda
RUN apt-get update
RUN apt-get install -y wget && \
        apt-get clean
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/anaconda.sh && \
        /bin/bash ~/anaconda.sh -b -p /opt/conda && \
        rm ~/anaconda.sh && \
        ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
        echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
        find /opt/conda/ -follow -type f -name '*.a' -delete && \
        find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
        /opt/conda/bin/conda clean -afy

# set path to conda
ENV PATH /opt/conda/bin:$PATH

# setup conda virtual environment
COPY environment.yml .
RUN conda install mamba -n base -c conda-forge
RUN mamba env create --name dclasso -f environment.yml
RUN R -e "IRkernel::installspec()"

RUN echo "conda activate dclasso" >> ~/.bashrc
ENV PATH /opt/conda/envs/dclasso/bin:$PATH
ENV CONDA_DEFAULT_ENV $dclasso

# set up project
RUN mkdir /dclasso
WORKDIR dclasso
EXPOSE 8888
