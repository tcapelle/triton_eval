FROM axolotlai/axolotl:main-latest

COPY ./axolotl_dev /app/axolotl_dev
WORKDIR /app/axolotl_dev

RUN pip install weave httpx

# Build with: docker build -t ghcr.io/tcapelle/triton_eval:latest .
# Push with: docker push ghcr.io/tcapelle/triton_eval:latest
