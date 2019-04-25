until ./trader; do
    echo "Trader crashed with exit code $?.  Respawning.." >&2
    sleep 1
done
