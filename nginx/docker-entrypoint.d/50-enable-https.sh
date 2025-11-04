#!/bin/sh
set -eu
(
  TARGET="/etc/nginx/conf.d/reverse-proxy.https.conf"
  DISABLED="/etc/nginx/conf.d/reverse-proxy.https.conf.disabled"
  FIRST_DOMAIN="api.chat-agente.arhook.com"
  if [ -n "$FIRST_DOMAIN" ]; then
    while [ ! -f "/etc/letsencrypt/live/$FIRST_DOMAIN/fullchain.pem" ]; do
      sleep 5
    done
    if [ -f "$DISABLED" ]; then
      cp "$DISABLED" "$TARGET"
    fi
    nginx -t && nginx -s reload || true
    while :; do
      sleep 21600
      nginx -s reload || true
    done
  fi
) &
