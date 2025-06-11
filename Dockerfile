FROM axolotlai/axolotl:main-latest

RUN pip install weave httpx wandb

# Copy everything to /app
COPY . /app

# Install the triton_eval package
RUN cd /app && pip install . 

# Set working directory
WORKDIR /app/axolotl_dev

# Download the fasttext model to the working directory
RUN curl -L -o lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

# Build with: docker build -t ghcr.io/tcapelle/triton_eval:latest .
# Push with: docker push ghcr.io/tcapelle/triton_eval:latest
