FROM axolotlai/axolotl:main-latest

RUN pip install weave httpx git+https://github.com/wandb/wandb.git@main

# Copy everything to /app
COPY . /app

# Install the triton_eval package
RUN cd /app && pip install . 

# Set working directory
WORKDIR /app/axolotl_dev

# Build with: docker build -t ghcr.io/tcapelle/triton_eval:latest .
# Push with: docker push ghcr.io/tcapelle/triton_eval:latest
