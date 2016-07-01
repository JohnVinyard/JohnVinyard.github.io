---
layout: post
title:  "Deploying a Search Interface with Docker Compose"
date:   2016-06-29 07:00:00 -0500
categories: zounds deploy
---

I'm late to the party, but I've been learning about Docker recently, specifically because I need a quick and easy way to deploy interactive examples for this blog.  [In a previous post](https://johnvinyard.github.io/zounds/indexing/search/2016/06/04/timbre-based-similarity-search.html), we built an interactive, timbre-based similarity search that you could [play with in your browser](http://23.253.119.81/).

# Docker Compose
I followed [this great example](https://developer.rackspace.com/blog/dev-to-deploy-with-docker-machine-and-compose/), and ended up with a [nice little repository](https://github.com/JohnVinyard/zounds-timbre-search) with everything needed to deploy the timbre-based similarity search to a rackspace server.  Most of the magic is in the `docker-compose.yaml` file, which describes the images and containers we'll need to run the service, as well as how they'll communicate with one another:

```yaml
version: '2'

services:
  zounds:
    build: ./zounds
    image: zounds

  timbre:
    build: ./timbre
    depends_on:
      - zounds
    ports:
      - "8888"

  nginx:
    build: ./nginx
    depends_on:
      - timbre
    links:
      - timbre:timbre
    ports:
      - "80:80"
```

First, we'll build a base image that installs [zounds](https://github.com/JohnVinyard/zounds).  Then, we'll build an image that will download a zip file containing the audio we'd like to index, process it, and start up a [tornado web server](http://www.tornadoweb.org/en/stable/).  Finally, we'll build an nginx image, and an nginx container that knows how to communicate with our tornado server.

# Docker Machine
The `deploy.sh` will provision a rackspace VM using `docker-machine`'s rackspace driver, and then use `docker-compose` to build the images from the `docker-compose.yaml` file, and start up the containers for the tornado and nginx web servers.  Before you run it, you'll need to export a few environment variables:

```bash
export OS_USERNAME=RACKSPACE_USERNAME
export OS_API_KEY=RACKSPACE_API_KEY
export OS_REGION_NAME=RACKSPACE_REGION_NAME
```

The `deploy.sh` script looks like this:

```bash
#!/usr/bin/env bash

echo "creating machine"
docker-machine create --driver rackspace --rackspace-flavor-id general1-2 timbre
echo docker-machine ip timbre

echo "securing machine"
docker-machine ssh timbre "apt-get update"
docker-machine ssh timbre "apt-get -y install fail2ban"
docker-machine ssh timbre "ufw default deny"
docker-machine ssh timbre "ufw allow ssh"
docker-machine ssh timbre "ufw allow http"
docker-machine ssh timbre "ufw allow 2376" # Docker
docker-machine ssh timbre "ufw --force enable"

eval "$(docker-machine env timbre)"
echo "building"
docker-compose build
echo "starting up"
docker-compose up -d
```

This toy example stores all of the data _inside_ the `timbre` container, which isn't what you'd normally want to do, because you'd lose that data any time you rebuilt the container, but I wanted to start with something really simple.
