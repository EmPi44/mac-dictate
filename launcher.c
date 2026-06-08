#include <unistd.h>
#include <spawn.h>
#include <signal.h>
#include <sys/wait.h>
extern char **environ;
static pid_t child = 0;
static void fwd(int s){ if (child) kill(child, s); }
int main(void){
    char *argv[] = {
        "/opt/homebrew/Caskroom/miniconda/base/bin/python3",
        "/Users/emir/repo_workspaces/mac-dictate/dictate_app.py",
        (void*)0
    };
    signal(SIGTERM, fwd);
    signal(SIGINT, fwd);
    if (posix_spawn(&child, argv[0], (void*)0, (void*)0, argv, environ) != 0) return 1;
    int st; while (waitpid(child, &st, 0) < 0);
    return 0;
}
