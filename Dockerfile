FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace
COPY . /workspace

RUN pip install --no-cache-dir -e .

ENTRYPOINT ["python", "-m"]
CMD ["midl.apps.fit", "--gin_file", "configs/experiment/main.gin"]
