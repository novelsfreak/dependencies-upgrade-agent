# Minimal egress-allowlist proxy for the sandbox's install phase.
# Built from alpine rather than pulled pre-built, so the allowlist is
# in this repo, in git, reviewable -- not trusted from a third-party tag.
FROM alpine:3.20
RUN apk add --no-cache squid
COPY squid.conf /etc/squid/squid.conf
EXPOSE 3128
CMD ["squid", "-N", "-f", "/etc/squid/squid.conf"]
