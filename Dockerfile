FROM ruby:latest
ENV DEBIAN_FRONTEND noninteractive

LABEL MAINTAINER Amir Pourmand

# Install dependencies, including Node.js for JavaScript runtime
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    locales \
    imagemagick \
    build-essential \
    zlib1g-dev \
    python3-pip \
    nodejs \
    npm \
    inotify-tools \
    procps && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* && \
    pip install nbconvert --break-system-packages

# Set up locales
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen

ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8 \
    JEKYLL_ENV=production

# Create the Jekyll working directory
RUN mkdir /srv/jekyll

# Add Gemfile and Gemfile.lock
ADD Gemfile.lock /srv/jekyll
ADD Gemfile /srv/jekyll

WORKDIR /srv/jekyll

# Install Jekyll and Bundler, and install gems
RUN gem install jekyll bundler
RUN bundle install --no-cache

# Expose the default Jekyll port
EXPOSE 8080

# Copy entry point script
COPY bin/entry_point.sh /tmp/entry_point.sh

# Set the entry point
CMD ["/tmp/entry_point.sh"]
