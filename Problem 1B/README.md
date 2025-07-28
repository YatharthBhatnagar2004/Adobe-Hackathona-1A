 BUID CMD -docker build -t adobe-hackathon-1b .
  RUN Commnad - docker run --rm -it \
    -v "$(pwd)/output:/app/output" \
    -v "$(pwd)/models:/app/models" \
    adobe-hackathon-1b
