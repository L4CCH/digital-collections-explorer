#!/usr/bin/env bash

set -euo pipefail

{

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-west-2}}"
SWITCH=""
BOOTSTRAP_URL=""
PUBLIC_URL=""
# Seconds to wait for the site to answer again after it restarts.
RESTART_TIMEOUT=300

fail() { echo "ERROR: $*" >&2; exit 1; }
source "$HERE/machine.sh"

[ $# -ge 1 ] || fail "usage: mcp.sh NAME --on|--off [--public-url URL] [--region REGION] [--bootstrap-url URL]"
NAME="$1"
shift
while [ $# -gt 0 ]; do
    case "$1" in
    --on) SWITCH="on"; shift ;;
    --off) SWITCH="off"; shift ;;
    --public-url) PUBLIC_URL="${2%/}"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --bootstrap-url) BOOTSTRAP_URL="$2"; shift 2 ;;
    *) fail "unknown option: $1" ;;
    esac
done
[ -n "$SWITCH" ] || fail "--on or --off is required"
[ -z "$PUBLIC_URL" ] || [[ "$PUBLIC_URL" =~ ^https?://[A-Za-z0-9.:/_~-]+$ ]] ||
    fail "--public-url must be an http:// or https:// address, such as https://library.example.org/explorer"
[ -z "$PUBLIC_URL" ] || [ "$SWITCH" = "on" ] || fail "--public-url only applies with --on"

STACK="dce-$NAME"
stack_value() {
    aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK" \
        --query "$1" --output text 2>/dev/null
}
instance="$(stack_value "Stacks[0].Outputs[?OutputKey=='InstanceId'].OutputValue")" ||
    fail "no deployment named $NAME in $REGION"
[ -n "$instance" ] && [ "$instance" != "None" ] || fail "no deployment named $NAME in $REGION"
site="$(stack_value "Stacks[0].Outputs[?OutputKey=='SiteUrl'].OutputValue")" || site=""
[ -n "$site" ] && [ "$site" != "None" ] || fail "could not read the address of $NAME"
if [ -z "$BOOTSTRAP_URL" ]; then
    BOOTSTRAP_URL="$(stack_value "Stacks[0].Parameters[?ParameterKey=='BootstrapUrl'].ParameterValue")" ||
        BOOTSTRAP_URL=""
fi
[ -n "$BOOTSTRAP_URL" ] && [ "$BOOTSTRAP_URL" != "None" ] ||
    BOOTSTRAP_URL="https://raw.githubusercontent.com/L4CCH/digital-collections-explorer/main/deploy/bootstrap.sh"
kept="$(stack_value "Stacks[0].Parameters[?ParameterKey=='PublicUrl'].ParameterValue")" || kept=""
[ "$kept" != "None" ] || kept=""

keep_public_url() {
    local keys key parameters=()
    keys="$(stack_value "Stacks[0].Parameters[].ParameterKey")" || keys=""
    case " $keys " in
    *[[:space:]]PublicUrl[[:space:]]*) ;;
    *) fail "this deployment was created before it could keep a public address. Keep it with: deploy/aws/deploy.sh $NAME --public-url $PUBLIC_URL --mcp --region $REGION" ;;
    esac
    for key in $keys; do
        [ "$key" = "PublicUrl" ] || parameters+=("ParameterKey=$key,UsePreviousValue=true")
    done
    echo "Keeping $PUBLIC_URL as the address in MCP links, for later runs"
    if ! aws cloudformation update-stack --region "$REGION" --stack-name "$STACK" --use-previous-template \
        --capabilities CAPABILITY_IAM --parameters "${parameters[@]}" \
        "ParameterKey=PublicUrl,ParameterValue=$PUBLIC_URL" >/dev/null ||
        ! aws cloudformation wait stack-update-complete --region "$REGION" --stack-name "$STACK"; then
        fail "could not keep $PUBLIC_URL in the deployment, so MCP was not switched. See: aws cloudformation describe-stack-events --stack-name $STACK --region $REGION"
    fi
}

if [ "$SWITCH" = "on" ]; then
    [ -z "$PUBLIC_URL" ] || [ "$PUBLIC_URL" = "$kept" ] || keep_public_url
    address="${PUBLIC_URL:-${kept:-$site}}"
    echo "Turning MCP on for $NAME (machine $instance), with links to $address"
    command="mcp --name $NAME --on --public-url $address"
else
    echo "Turning MCP off for $NAME (machine $instance)"
    command="mcp --name $NAME --off"
fi
run_on_machine_when_ready \
    "$(bootstrap_commands "$command")" \
    "the machine did not switch MCP. If it runs an older version, run deploy/aws/update.sh first." \
    "deploy/aws/mcp.sh $NAME --$SWITCH --region $REGION"

[ "$SWITCH" = "on" ] || { echo "Done. Once the site has restarted, it no longer answers at /mcp."; exit 0; }

echo "The site is restarting. Waiting for $address/mcp to answer."
SECONDS=0
while :; do
    answer="$(curl -s -o /dev/null -w '%{http_code}' -X POST "$address/mcp" \
        -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' -d '{}')" ||
        answer="000"
    case "$answer" in
    2?? | 400 | 405 | 406) break ;;
    404)
        health="$(curl -s -o /dev/null -w '%{http_code}' "$address/api/health")" || health="000"
        [ "$health" = "200" ] &&
            fail "the site is up but does not answer at /mcp, so it runs a version without MCP. Switch it to a newer image with deploy/aws/update.sh $NAME --image IMAGE, then run this again."
        ;;
    421)
        fail "the site turned away $address/mcp, since the request reached it under a name it does not know. If CloudFront or a proxy of your own forwards to it, run: deploy/aws/configure.sh $NAME --set proxy_hops=1 --region $REGION"
        ;;
    esac
    [ "$SECONDS" -lt "$RESTART_TIMEOUT" ] ||
        fail "the site did not answer at $address/mcp within $RESTART_TIMEOUT seconds. See deploy/aws/status.sh $NAME --region $REGION"
    sleep 10
done

echo "Done. AI assistants can connect to $address/mcp, for example:"
echo "  claude mcp add --transport http $NAME $address/mcp"
if [ "$address" = "$site" ]; then
    case "$site" in
    http://*) echo "Assistants that connect from the cloud, such as Claude's custom connectors, need https://. Deploy with --https for that." ;;
    esac
fi

exit 0
}
