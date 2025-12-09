import resource
 
def get_unix_thread_limit():
    # Get soft and hard limits for RLIMIT_NPROC (process/thread count)
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NPROC)
    print(f"Per-user thread limit (soft): {soft_limit}")
    print(f"Per-user thread limit (hard): {hard_limit}")
    return soft_limit
 
get_unix_thread_limit()