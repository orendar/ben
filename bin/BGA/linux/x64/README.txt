BGADLL.so  BGADLL v0.9.9.3 release zip (github.com/ThorvaldAagaard/BGA), linux/x64.
libdds.so  github.com/dds-bridge/dds tag v3.1.0, library/src compiled with:
             g++ -O3 -std=c++20 -fPIC -pthread -DNDEBUG -w -I. -I<inc> -c, linked -shared
           (<inc> holds a symlink dds -> library/src).

The libdds shipped before this rejected every thread index above 0 with -15, and
BGADLL scored those solves as 0 tricks, so ~7/8 of PIMC's samples were zeros and
which ones depended on the thread race. v3.1.0 ignores the index and is reentrant.
With these two, Ben's card play is a pure function of its query.
