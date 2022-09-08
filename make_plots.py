import os

import matplotlib.pyplot as plt
import numpy as np
import pydicom as pdcm

from utility import rle_decode_modified, mask2rle, rle_decode


def make_plots():
    pneumo = "606355 2 1020 8 1015 12 1011 12 1010 14 1009 15 1008 16 1007 17 1006 18 1005 19 1004 20 1003 21 1003 20 1003 21 1003 21 1002 22 1002 21 1002 22 1002 22 1001 23 1001 22 1001 23 1001 23 1001 23 1000 24 1000 23 1000 24 1000 24 1000 24 999 24 1000 24 1000 24 999 25 999 24 1000 24 999 25 999 25 999 24 1000 24 1000 24 1000 24 1000 24 1000 23 1000 24 1000 24 1000 24 1000 24 1000 24 1000 24 1000 24 1000 24 1000 24 999 25 999 25 999 25 999 25 999 25 999 25 999 25 999 25 999 25 999 25 999 25 999 25 999 25 999 25 999 26 998 26 998 26 998 26 999 25 999 26 998 26 998 26 998 26 998 27 997 27 997 27 997 27 998 27 997 27 997 27 997 27 998 27 997 27 997 28 996 28 997 28 996 28 996 29 995 29 996 29 995 29 995 29 995 30 995 29 995 30 994 30 995 29 995 30 995 29 995 30 994 30 995 30 994 31 994 30 995 30 994 30 995 30 995 29 995 30 995 29 996 29 995 29 996 29 996 29 995 29 996 29 996 29 995 29 996 29 996 29 995 29 996 29 995 29 996 29 996 28 996 29 996 28 997 28 996 28 997 27 998 27 998 26 998 27 998 26 999 25 999 26 999 25 1000 25 1000 24 1001 24 1001 23 1001 24 1001 23 1002 23 1002 22 1003 21 1004 21 1004 20 1005 20 1005 19 1006 18 1008 17 1008 16 1009 16 1010 14 1011 14 1011 14 1012 12 1013 12 1013 12 1013 11 1014 11 1014 11 1015 9 1016 9 1016 9 1016 9 1016 9 1016 9 1016 9 1016 9 1016 9 1016 10 1015 10 1015 10 1015 11 1014 11 1014 12 1013 12 1014 12 1013 13 1012 13 1013 13 1013 12 1014 11 1015 11 1015 10 1015 10 1016 9 1016 10 1015 10 1016 9 1016 10 1016 9 1016 9 1016 10 1016 9 1016 10 1016 10 1015 11 1014 12 1014 12 1013 13 1012 14 1011 16 1010 16 1009 18 1007 19 1006 20 1006 20 1006 20 1006 19 1007 19 1007 18 1008 18 1008 18 1008 18 1009 17 1009 18 1009 18 1009 17 1009 18 1009 20 1007 24 1002 26 1001 27 1000 26 1000 27 1000 27 1000 26 1000 27 1000 26 1001 25 1001 25 1003 23 1005 20 1008 18 1010 16 1013 13 1015 10"
    rib1 = "768197 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60 964 60"
    rib2 = "778505 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69 955 69"
    rib3 = "780636 64 960 64 960 64 960 64 960 64 960 64 960 64 960 64 960 64 960 64 960 64 960 64 960 64 960 64 960 64 960 64 960 64 960 64 960 64 960 64 960 64 960 64 960 64 960 64 960 64 960 64 960 64 960 64 960 64 960 64 960 64 960 64 960 64 960 64 960 64 960 64 960 64 960 64 960 64 960 64 960 64 960 64 960 64 960 64"
    tube = "795213 19 1004 1 18 1 1004 1 18 1 1003 1 19 1 1003 1 19 1 1002 1 20 1 1002 1 20 1 1001 1 21 1 1001 1 21 1 1000 1 22 1 1000 1 22 1 999 1 23 1 999 1 23 1 998 1 24 1 998 1 24 1 997 1 25 1 997 1 25 1 996 1 26 1 996 1 26 1 995 1 27 1 995 1 27 1 994 1 28 1 994 1 28 1 994 1 28 1 994 1 28 1 994 1 27 1 995 1 26 1 996 1 25 1 997 1 25 1 997 1 24 1 998 1 23 1 999 1 22 1 1000 1 21 1 1001 1 20 1 1002 1 20 1 1002 1 19 1 1003 1 18 1 1004 1 17 1 1005 1 16 1 929 2 75 1 15 1 932 4 71 1 15 1 936 4 67 1 14 1 941 4 63 1 13 1 946 4 59 1 12 1 951 4 55 1 11 1 956 4 51 1 10 1 961 4 48 1 9 1 965 4 44 1 8 1 970 3 41 1 7 1 974 2 39 1 5 2 977 2 37 1 3 2 981 1 37 3 984 2 34 2 988 2 29 3 1 1 990 1 26 2 4 1 991 2 22 2 6 1 993 2 18 2 9 1 994 2 13 3 11 1 996 1 10 2 14 1 997 2 6 2 16 1 999 2 2 2 18 1 1001 2 21 1 1023 1 1023 1 1023 1 1023 1 1024 1 1023 1 1023 1 1023 1 1023 1 1024 1 1023 1 1023 1 1023 1 1023 1 1024 1 1023 1 1023 1 1023 1 1023 1 1023 1 1024 1 1023 1 1023 1 1023 1 1023 1 1024 1 1023 1 1023 1 1023 1 1024 1 1023 1 1023 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1023 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1023 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 3 1024 4 1024 5 1024 4 1024 5 1024 5 1024 4 1024 5 1024 5 1024 6 1024 5 1024 5 1024 5 1024 6 1024 5 1024 6 1024 6 1024 6 1024 7 1024 6 1024 6 1024 7 1024 6 1024 6 1024 6 1024 5 1024 5 1024 5 1024 6 1024 5 1024 5 1024 5 1024 5 1024 5 1024 5 1024 5 1024 6 1024 5 1024 5 1024 5 1024 5 1024 5 1024 5 1024 5 1024 6 1024 5 1024 5 1024 5 1024 5 1024 5 1024 5 1024 6 1024 5 1024 5 1024 5 1024 5 1024 5 1024 5 1024 5 1024 6 1024 5 1024 5 1024 5 1024 3"


    test1 = "208490 6 965 17 27 26 944 85 937 97 130 14 781 104 70 73 775 111 58 82 770 120 47 90 763 265 757 269 752 274 747 277 743 282 737 286 734 290 730 293 728 295 727 297 724 299 723 300 722 302 720 303 720 303 720 302 720 303 719 304 718 305 717 306 716 307 715 308 715 309 713 310 712 311 710 313 709 315 707 316 706 317 706 317 705 318 704 320 702 321 702 321 701 323 699 324 698 326 696 327 695 329 693 331 691 333 689 334 688 336 686 337 685 339 682 341 681 343 679 345 677 347 675 349 672 351 669 354 668 356 665 358 662 361 660 364 658 366 656 368 653 370 651 372 651 373 649 374 647 376 645 379 644 379 643 381 642 381 641 382 640 384 638 385 637 386 636 388 634 390 631 393 630 393 630 394 628 395 627 397 625 398 625 399 623 400 623 401 622 401 621 402 621 402 620 404 619 404 619 403 620 402 621 401 622 400 623 399 624 398 625 398 625 397 626 397 626 397 626 396 627 396 627 395 628 394 629 394 630 392 631 392 631 391 633 389 634 389 634 389 634 390 634 389 634 389 634 389 634 390 634 389 634 389 634 388 635 388 636 387 636 387 636 387 637 386 637 386 637 386 638 386 637 386 638 386 637 386 637 387 636 388 636 387 636 387 636 388 636 387 637 387 636 387 637 386 637 387 637 386 637 386 637 386 638 385 638 385 639 385 638 385 639 384 639 385 638 385 639 384 639 384 639 383 641 382 642 381 643 381 643 381 643 381 643 380 643 380 644 379 644 379 645 378 646 378 645 379 645 379 644 380 644 379 645 377 647 376 648 373 650 370 654 367 657 366 658 366 658 365 659 364 660 363 660 363 661 362 662 361 662 361 663 360 663 359 665 358 666 357 667 355 668 354 670 352 672 350 674 349 675 348 675 348 676 347 677 346 678 344 680 342 681 341 683 340 684 338 686 337 687 335 688 334 690 332 692 330 694 327 697 322 702 314 710 313 711 312 712 309 715 304 720 301 723 299 725 295 729 290 734 288 736 286 738 285 739 284 740 282 742 281 743 280 744 280 744 279 745 279 745 278 746 277 747 277 747 276 748 275 749 274 750 274 749 274 750 273 751 271 753 270 754 269 755 268 757 266 758 265 759 263 761 262 763 260 764 260 765 258 766 257 767 257 768 255 769 255 769 255 769 255 769 254 770 253 772 251 773 251 773 250 774 249 775 248 776 248 777 246 778 245 779 245 779 244 781 242 782 241 784 240 784 239 786 237 787 236 788 236 788 235 790 233 791 232 793 230 795 228 796 227 798 226 798 225 799 224 800 223 802 222 802 221 804 219 806 217 807 217 808 215 809 214 811 213 811 213 812 212 812 211 814 209 815 209 816 207 818 205 819 205 820 204 820 204 820 203 821 202 822 202 822 201 823 200 823 200 824 200 823 200 824 199 824 200 823 199 825 197 826 195 831 191 835 187 839 183 843 179 852 169 857 165 863 159 875 147 882 135 895 127 903 117 916 105 929 93 941 79 951 68 966 52 978 40"
    test2 = "218477 3 1016 6 1016 5 1016 6 1015 7 1013 9 1011 11 1011 11 1010 12 1010 12 1010 12 1010 12 1010 12 1010 12 1010 12 1010 12 1010 13 1008 14 1008 15 1007 16 1006 16 1006 17 1005 17 1005 18 1004 19 1003 19 1003 20 1002 20 1002 21 1001 22 1000 22 1000 23 999 24 998 24 998 25 997 26 997 26 996 27 995 28 994 28 994 29 993 30 992 31 991 32 991 32 990 33 989 34 988 35 987 36 986 37 986 37 986 37 985 38 985 38 984 39 984 39 984 39 983 40 983 40 983 40 983 40 983 40 983 41 982 41 982 41 982 42 981 42 980 43 980 43 980 43 980 43 980 44 979 44 980 43 980 43 980 43 980 44 979 44 979 44 980 44 979 44 979 44 979 44 979 45 978 45 979 44 979 45 978 45 979 44 979 44 979 45 978 45 979 44 979 45 978 45 979 44 979 45 978 45 978 45 979 45 978 45 978 46 978 45 978 45 978 46 977 46 978 45 978 46 977 46 978 45 978 46 977 46 978 46 977 46 978 46 977 46 978 46 977 46 978 45 978 46 978 45 978 45 979 45 978 45 978 45 979 45 978 45 979 45 979 44 979 45 979 44 979 45 979 44 980 44 979 44 980 43 981 43 980 43 981 43 980 43 981 43 981 42 981 43 981 42 982 42 982 41 983 41 982 41 983 41 983 41 983 40 983 41 983 40 984 40 984 40 984 39 985 39 985 39 985 38 985 39 985 38 986 38 986 38 986 37 987 37 987 37 987 36 988 36 988 35 989 35 989 35 989 34 990 34 990 34 990 34 990 34 990 34 990 34 990 34 990 34 990 34 990 34 990 34 990 34 990 34 990 34 990 34 990 34 990 33 991 33 991 33 991 32 992 32 992 31 993 31 994 30 994 29 995 29 995 29 995 28 996 28 996 27 997 27 998 26 998 26 998 26 998 26 999 25 999 25 999 25 999 25 999 25 1000 24 1000 24 1001 23 1001 23 1002 23 1001 23 1002 22 1002 22 1003 22 1002 22 1003 21 1003 22 1002 22 1003 22 1002 22 1003 21 1003 22 1002 22 1003 22 1002 22 1003 22 1002 22 1002 23 1002 22 1002 23 1002 23 1001 24 1001 24 1000 25 1000 25 999 26 999 26 998 27 998 26 998 27 998 27 997 28 997 28 996 29 996 29 996 28 997 28 996 29 996 29 996 29 996 29 996 29 996 29 996 29 996 29 996 29 996 29 996 29 997 28 998 27 999 26 1000 24 1003 21 1005 19 1009 15 1013 11 1017 7 1021 3"

    #pneumo1 = "221609 13 1002 21 995 28 987 37 979 44 971 52 966 56 964 59 958 65 953 69 951 71 948 74 946 76 944 78 942 81 937 85 932 91 929 94 926 96 924 99 920 102 918 105 915 108 914 108 913 109 912 110 910 112 908 114 908 114 907 115 906 116 906 115 906 115 906 116 906 115 906 115 906 115 906 115 906 115 906 116 906 115 906 115 906 116 906 115 906 116 905 117 905 117 904 118 904 117 905 117 905 116 906 115 908 114 908 114 908 114 908 114 908 114 908 114 908 113 909 113 909 112 910 112 910 112 910 112 911 111 911 111 912 110 913 109 913 109 914 108 914 108 915 107 916 107 915 107 916 106 916 106 917 106 917 105 917 105 918 104 919 104 919 103 920 103 920 103 920 102 921 102 921 102 921 102 920 102 921 102 921 102 921 102 921 101 922 101 922 101 922 100 923 100 923 100 923 100 924 99 924 99 924 99 925 98 925 98 925 99 924 99 925 98 925 99 924 99 925 98 925 98 925 98 925 98 926 98 925 98 925 98 926 97 926 97 926 97 927 97 926 97 927 96 927 96 928 95 928 96 928 95 928 95 929 95 929 94 929 94 930 93 930 93 931 92 932 92 931 92 932 91 933 90 934 89 934 90 934 89 935 88 936 88 935 88 936 87 937 87 936 87 937 86 937 87 937 86 938 86 937 86 938 85 939 85 939 84 940 83 941 83 941 82 942 81 943 80 943 81 944 79 945 78 946 78 946 77 947 76 948 76 948 76 948 75 950 74 950 73 952 72 952 72 953 70 954 70 955 69 955 69 956 67 958 66 959 65 960 64 961 63 962 62 963 61 965 59 966 58 967 57 969 55 971 53 973 51 975 48 978 46 981 43 985 39 991 32 1000 24 1009 15 1020 4"
    #pneumo2 = "373863 11 1009 17 1003 22 1000 26 996 30 992 33 989 35 988 37 985 39 983 42 981 44 979 45 978 47 975 49 974 50 973 52 971 53 971 54 969 55 968 56 968 57 966 58 965 60 964 60 964 61 963 61 963 62 962 62 962 63 961 63 961 63 960 64 961 64 960 64 960 64 960 64 960 65 959 65 959 65 959 65 960 64 960 64 960 64 961 63 961 63 962 63 961 63 961 63 962 62 963 61 963 61 964 60 965 59 965 59 966 57 968 56 969 55 969 54 971 53 972 51 973 51 974 50 975 48 978 44 981 42 983 40 986 36 990 31 996 24 1002 15 1011 5"
    #pneumo1 = rle_decode_modified(pneumo1, (1024, 1024))
    #pneumo2 = rle_decode_modified(pneumo2, (1024, 1024))
    #pneumo1 = rle_decode(pneumo1, (1024, 1024))
    #pneumo2 = rle_decode(pneumo2, (1024, 1024))

    pn1 = "250542 11 1011 14 1009 15 1008 17 1005 20 1003 22 1002 21 1002 22 1002 21 1002 22 1001 22 1002 22 1001 22 1002 22 1002 21 1003 21 1003 20 1004 19 1005 19 1005 18 1005 19 1006 17 1007 17 1007 17 1007 16 1009 15 1009 15 1009 14 1010 14 1010 14 1011 12 1013 11 1013 11 1014 10 1015 9 1015 9 1016 8 1020 4"
    pn2 = "250542 11 1011 14 1009 15 1008 17 1005 20 1003 22 1002 21 1002 22 1002 21 1002 22 1001 22 1002 22 1001 22 1002 22 1002 21 1003 21 1003 20 1004 19 1005 19 1005 18 1005 19 1006 17 1007 17 1007 17 1007 16 1009 15 1009 15 1009 14 1010 14 1010 14 1011 12 1013 11 1013 11 1014 10 1015 9 1015 9 1016 8 1020 4"
    pn3 = "192189 13 1008 21 1002 25 997 29 994 32 992 33 991 34 1 1 988 36 988 36 988 36 988 37 987 37 987 37 987 38 987 37 988 37 987 37 988 36 989 35 990 34 991 33 994 30 996 28 998 26 999 25 999 25 999 25 999 25 999 25 998 26 997 27 997 27 996 28 995 29 995 28 996 28 995 28 996 28 995 28 996 28 996 27 997 27 996 27 997 26 997 27 996 27 997 26 997 25 998 25 998 24 999 23 1000 23 1001 21 1002 21 1002 20 1003 19 1005 17 1006 16 1007 15 1009 14 1009 14 1010 13 1010 14 1010 13 1011 13 1010 14 1010 13 1010 13 1011 13 1011 12 1011 13 1011 12 1011 13 1011 13 1011 12 1012 12 1012 11 1013 11 1013 11 1013 11 1013 11 1013 11 1014 10 1014 10 1014 10 1014 10 1014 10 1015 9 1015 9 1015 8 1016 8 1016 8 1016 7 1017 7 1017 6 1018 5 1022 2"
    pn4 = "201381 2 1021 3 1020 5 1018 6 1017 7 1016 9 1014 10 1015 10 1014 10 1014 10 1014 10 1015 9 1015 9 1015 9 1015 9 1020 4"
    pn5 = "166643 8 1011 18 1004 22 1000 26 996 30 991 34 988 36 985 40 982 43 979 47 975 50 972 53 970 56 966 59 963 60 962 12 21 29 959 11 28 26 955 11 34 24 952 11 39 21 957 5 42 20 1005 19 1005 19 1006 17 1008 15 1008 16 1007 16 1008 15 1008 16 1007 16 1007 16 1007 16 1007 17 1006 17 1007 16 1007 17 1006 17 1006 17 1007 16 1007 17 1006 17 1007 16 1007 17 1006 17 1007 16 1007 16 1008 15 1008 15 1009 14 1009 14 1009 14 1010 13 1010 13 1011 12 1012 11 1013 10 1013 10 1014 9 1015 8 1016 7 1016 7 1020 3"
    pn1 = rle_decode_modified(pn1, (1024, 1024))
    pn2 = rle_decode_modified(pn2, (1024, 1024))
    pn3 = rle_decode_modified(pn3, (1024, 1024))
    pn4 = rle_decode_modified(pn4, (1024, 1024))
    pn5 = rle_decode_modified(pn5, (1024, 1024))

    maskToTry = "325152 5 326171 16 327194 18 328217 20 329239 23 330262 24 331284 27 332306 30 333327 33 334347 38 335370 39 336392 42 337415 43 338438 44 339460 46 340483 47 341505 49 342529 49 343553 49 344576 50 345600 50 346624 50 347648 50 348672 50 349696 51 350720 51 351744 51 352768 51 353793 50 354817 50 355841 50 356865 50 357889 50 358913 50 359939 48 360964 47 361990 45 363016 43 364042 41 365056 6 365067 40 366074 16 366093 38 367095 21 367119 36 368115 28 368144 36 369138 66 370160 68 371182 70 372205 71 373227 73 374250 74 375274 74 376297 75 377320 76 378343 77 379365 79 380388 80 381410 82 382432 84 383454 86 384476 88 385497 91 386517 95 387538 98 388554 106 389575 109 390596 112 391615 117 392639 117 393663 117 394686 118 395710 118 396734 118 397758 118 398782 118 399806 118 400830 118 401854 118 402877 119 403901 119 404924 120 405948 120 406972 119 407996 119 409020 119 410044 118 411068 117 412092 116 413116 115 414140 114 415163 114 416187 114 417211 113 418235 113 419258 114 420282 114 421305 115 422329 115 423353 115 424377 114 425400 115 426424 115 427448 115 428471 115 429494 116 430514 120 431534 123 432556 125 433577 128 434599 130 435622 130 436645 131 437668 132 438691 133 439715 132 440738 133 441762 133 442786 133 443810 132 444833 133 445857 132 446881 130 447905 129 448928 129 449952 127 450976 126 452000 122 453024 122 454048 122 455072 122 456096 122 457120 122 458144 122 459168 122 460192 122 461216 123 462240 123 463264 123 464288 123 465312 124 466336 124 467360 125 468384 125 469408 125 470432 125 471456 125 472480 125 473504 125 474528 125 475552 125 476576 124 477599 125 478623 125 479647 124 480671 124 481695 123 482718 123 483742 122 484766 121 485790 120 486814 119 487837 120 488861 119 489885 118 490909 117 491933 117 492957 116 493981 113 495005 111 496029 110 497053 109 498077 108 499101 107 500125 106 501149 106 502173 106 503197 106 504221 106 505245 106 506269 106 507293 106 508317 106 509341 107 510365 107 511389 108 512413 108 513437 109 514461 110 515485 111 516509 112 517533 113 518557 114 519581 115 520605 116 521629 117 522653 118 523677 119 524701 120 525725 121 526749 121 527773 122 528797 123 529821 123 530845 124 531869 124 532893 125 533917 125 534941 125 535965 126 536989 126 538013 126 539037 126 540061 126 541085 126 542109 126 543133 126 544157 126 545181 126 546205 126 547229 126 548253 126 549277 126 550301 126 551325 126 552349 126 553373 126 554397 126 555421 126 556445 126 557469 126 558493 126 559517 126 560541 126 561565 125 562589 125 563613 125 564637 125 565661 124 566685 124 567709 124 568733 124 569757 124 570782 123 571806 123 572830 123 573854 123 574878 123 575903 121 576927 121 577951 120 578975 120 579999 120 581023 119 582047 119 583072 118 584096 117 585120 117 586144 117 587169 115 588193 115 589218 114 590242 113 591267 112 592291 112 593316 110 594340 110 595364 109 596388 109 597413 107 598437 106 599461 106 600486 104 601510 103 602535 101 603560 99 604585 98 605609 97 606634 91 607659 84 608683 72 609707 70 610732 65 611756 63 612781 61 613806 55 614831 48 615855 46 616880 44 617905 42 618929 40 619954 38 620978 37 622003 36 623028 34 624053 33 625078 32 626104 29 627129 28 628154 26 629180 24 630205 22 631235 16 632266 8 633262 3 633293 5 634280 21 634319 2 635300 26 636323 29 637345 32 638368 35 639390 38 640412 42 641433 47 642457 50 643480 52 644504 53 645527 55 646551 56 647574 58 648598 60 649621 62 650646 62 651670 63 652695 63 653719 64 654746 63 655773 61 656801 56 657828 53 658855 45 659882 38 667103 1 668122 13 669144 19 670167 26 671189 30 672212 1 672219 26 673231 40 674253 44 675275 48 676296 53 677318 57 678339 62 679360 66 680382 68 681403 72 682427 73 683451 73 684475 74 685499 43 685548 25 686525 39 686572 25 687550 35 687596 26 688577 30 688620 26 689604 24 689644 26 690639 6 690668 26 691692 26 692716 26 693740 26 694764 26 695788 26 696812 25 697836 25 698859 26 699883 26 700907 26 701930 27 702953 27 703976 28 704999 29 706022 30 707045 30 708068 31 709091 32 710114 32 711137 32 712160 33 713183 33 714205 28 715227 27 716255 20"
    maskToTry = "501985 7 503008 13 504031 16 505054 19 506077 20 507101 20 508124 21 509147 22 510170 23 511193 24 512216 25 513239 26 514262 27 515286 27 516309 28 517332 29 518355 30 519378 31 520401 32 521424 33 522447 34 523471 34 524494 35 525517 36 526540 38 527563 40 528587 41 529611 42 530635 44 531659 45 532683 47 533706 53 534730 57 535754 62 536778 65 537801 69 538825 73 539849 75 540872 79 541896 80 542920 81 543944 82 544967 84 545991 84 547015 85 548038 86 549062 87 550085 88 551109 89 552132 90 553156 90 554180 91 555203 92 556227 91 557251 89 558274 89 559298 70 560322 67 561345 67 562369 66 563393 64 564417 62 565441 60 566465 59 567489 57 568513 56 569537 55 570561 54 571585 52 572609 51 572684 16 573633 49 573698 31 574657 48 574720 35 575681 46 575742 39 576705 44 576765 42 577729 43 577787 47 578753 42 578810 49 579778 40 579832 51 580804 37 580853 55 581829 35 581876 57 582856 30 582898 60 583883 25 583921 63 584944 65 585967 67 586991 67 588014 69 589038 70 590062 71 591087 72 592111 73 593136 73 594160 75 595184 76 596208 77 597232 79 598256 80 599280 80 600304 81 601327 82 602351 83 603374 85 604398 85 605422 86 606446 86 607470 87 608493 88 609517 89 610541 89 611565 90 612590 89 613614 90 614639 91 615663 92 616688 93 617713 93 618737 95 619762 99 620787 100 621812 102 622836 104 623861 106 624886 22 624936 56 625911 12 625966 8 625982 35 626936 1 627009 33 628037 30 629064 27 630090 25 631115 24 632141 22 633166 21 634192 17 635218 14 636248 6"
    maybeThisOne = "413905 1 414928 2 415952 1 416975 2 417999 1 419022 1 420046 1 421069 1 422093 1 423116 1 424140 1 425163 1 426186 2 427210 1 428233 2 429256 2 430279 3 431303 2 432326 3 433350 2 434373 3 435397 3 436420 3 437444 3 438467 4 439491 3 440514 4 441538 3 442561 4 443585 4 444609 3 445632 4 446656 4 447679 5 448703 5 449727 5 450750 6 451774 6 452797 7 453821 7 454844 8 455868 8 456891 9 457915 9 458938 10 459962 10 460986 10 462009 11 463033 11 464056 12 465080 12 466104 12 467127 13 468151 12 469174 13 470198 13 471221 14 472245 14 473268 15 474292 15 475315 16 476339 16 477363 15 478386 16 479410 16 480433 17 481457 17 482481 17 483504 18 484528 18 485552 17 486575 18 487599 18 488622 19 489646 19 490670 19 491693 20 492717 20 493741 20 494764 20 495788 20 496811 21 497835 21 498859 21 499882 22 500906 22 501930 22 502953 22 503977 22 505000 23 506024 23 507048 23 508071 24 509095 24 510119 24 511142 25 512166 24 513189 25 514213 25 515237 25 516260 26 517284 26 518308 25 519331 26 520355 26 521378 27 522402 26 523426 26 524450 26 525473 27 526497 27 527521 26 528545 26 529568 27 530592 27 531616 26 532640 26 533664 26 534688 26 535711 26 536735 26 537759 25 538783 25 539807 25 540831 24 541855 24 542878 25 543902 24 544926 24 545950 23 546973 24 547997 24 549021 23 550045 23 551069 23 552093 22 553116 23 554140 22 555164 22 556188 22 557212 21 558236 21 559259 21 560283 21 561307 20 562330 21 563354 20 564377 21 565401 20 566425 19 567449 19 568472 19 569496 19 570520 18 571544 18 572567 18 573591 18 574615 17 575638 18 576662 17 577686 17 578709 17 579733 17 580756 17 581780 17 582804 16 583828 16 584851 16 585875 16 586899 15 587923 15 588946 15 589970 15 590994 14 592018 14 593041 14 594065 14 595089 14 596112 14 597136 14 598159 14 599183 14 600207 13 601231 13 602254 13 603278 13 604302 12 605326 12 606349 13 607373 12 608397 12 609421 11 610444 12 611468 12 612492 11 613516 11 614539 11 615563 11 616587 10 617611 10 618635 9 619658 10 620682 9 621706 9 622730 9 623753 9 624777 9 625801 8 626825 8 627848 9 628872 8 629896 8 630920 8 631943 8 632967 8 633991 8 635015 8 636039 7 637063 7 638086 8 639110 8 640134 8 641158 7 642182 7 643206 7 644230 7 645253 7 646277 7 647301 7 648325 7 649348 7 650372 7 651396 7 652420 7 653444 6 654468 6 655491 7 656515 7 657539 7 658563 7 659587 7 660611 6 661635 6 662659 6 663683 6 664706 7 665730 7 666754 7 667778 7 668802 7 669826 7 670850 7 671874 7 672897 8 673921 8 674945 8 675969 8 676993 8 678017 7 679041 7 680065 7 681089 7 682113 7 683137 7 684161 7 685185 7 686209 7 687233 7 688257 7 689281 7 690305 7 691329 7 692353 7 693377 6 694401 6 695425 6 696449 6 697473 5 698497 5 699521 5 700545 5 701569 4 702593 4 703617 4 704642 3 705666 2 706691 1 888944 1 889967 3 890991 3 892014 5 893038 6 894062 6 895085 8 896109 9 897133 10 898156 11 899180 12 900203 14 901227 15 902251 16 903275 17 904298 19 905322 21 906346 22 907370 24 908393 28 909417 30 910441 32 911465 34 912489 37 913512 47 914536 48 915560 48 916583 50 917607 50 918630 52 919654 52 920678 53 921702 54 922725 55 923749 56 924773 56 925797 57 926820 59 927844 59 928868 60 929892 61 930916 61 931939 63 932963 63 933987 64 935011 65 936035 65 937059 66 938083 66 939107 67 940131 67 941155 68 942179 68 943203 69 944228 68 945252 67 946277 66 947301 66 948326 64 949351 61 950376 59 951401 57 952426 55 953450 16 953479 25 954475 6 954513 6"
    maskToTry = "211549 10 212569 22 213589 35 214609 43 215629 51 216651 62 217673 68 218695 74 219717 80 220740 86 221762 91 222785 94 223807 99 224830 103 225853 106 226875 111 227898 115 228920 119 229943 123 230966 125 231989 128 233012 130 234035 133 235058 135 236081 138 237104 141 238128 143 239151 146 240174 148 241197 151 242221 152 243244 155 244268 156 245291 158 246315 160 247338 162 248362 164 249385 166 250409 167 251432 170 252456 171 253480 173 254503 175 255527 176 256551 178 257575 179 258598 182 259622 183 260646 184 261670 186 262694 187 263718 189 264741 115 264881 51 265765 111 265909 48 266789 107 266937 46 267813 104 267966 42 268837 102 268994 40 269861 99 270021 38 270885 97 271047 37 271908 96 272074 36 272932 94 273101 34 273956 92 274127 34 274980 91 275154 32 276003 90 276181 30 277027 89 277207 29 278051 87 278234 28 279075 86 279260 27 280099 85 280286 26 281123 83 281311 26 282146 83 282337 25 283170 81 283362 25 284194 80 284388 25 285218 79 285413 25 286242 77 286439 24 287266 76 287464 24 288290 74 288490 23 289314 73 289515 23 290338 72 290541 23 291361 71 291566 23 292385 70 292592 22 293409 68 293617 22 294433 67 294642 22 295457 66 295668 22 296481 65 296693 22 297505 64 297718 22 298529 63 298743 22 299553 62 299768 22 300576 62 300794 21 301600 60 301819 20 302624 58 302844 20 303648 56 303869 20 304672 54 304894 20 305696 51 305919 20 306720 49 306945 19 307744 46 307970 18 308768 45 308995 18 309792 43 310020 18 310816 42 311045 18 311840 40 312071 17 312864 39 313096 16 313888 38 314121 16 314912 37 315146 16 315936 35 316171 16 316960 34 317196 15 317984 33 318221 15 319008 32 319246 15 320032 31 320271 14 321056 29 321296 14 322080 28 322321 14 323104 27 323345 15 324128 26 324370 14 325152 25 325395 14 326176 24 326420 14 327200 23 327444 14 328224 22 328469 14 329248 21 329494 13 330272 20 330518 14 331296 18 331543 14 332320 16 332568 13 333345 13 333593 13 334369 11 334617 13 335394 9 335642 13 336418 7 336667 12 337443 5 337692 11 338467 3 338717 10 339492 1 339742 10 340767 9 341792 1 435402 1 436425 10 437449 12 438472 14 439495 17 440518 20 441542 22 442565 25 443589 27 444612 30 445636 33 446660 35 447683 39 448707 41 449730 44 450754 46 451778 48 452802 50 453825 53 454849 55 455873 57 456897 58 457920 61 458944 62 459968 64 460992 65 462016 66 463040 67 464063 69 465087 70 466111 71 467135 72 468159 73 469183 73 470207 74 471231 75 472255 76 473278 77 474302 78 475326 79 476350 79 477374 80 478398 81 479421 83 480445 83 481469 84 482493 85 483516 86 484540 87 485564 87 486588 88 487612 88 488635 90 489659 90 490683 91 491707 91 492730 93 493754 93 494778 94 495801 95 496825 96 497849 96 498872 98 499896 98 500919 100 501943 100 502967 101 503991 101 505014 103 506038 103 507062 103 508086 104 509109 105 510133 105 511157 105 512181 106 513205 106 514228 107 515252 107 516276 108 517300 108 518323 109 519347 109 520371 110 521395 110 522418 111 523442 111 524466 111 525490 111 526513 111 527537 111 528561 111 529585 111 530609 111 531633 111 532656 112 533680 112 534704 112 535728 111 536752 111 537776 111 538800 111 539823 112 540847 112 541871 111 542895 111 543918 111 544942 111 545966 111 546990 110 548013 111 549037 111 550061 110 551085 110 552108 111 553132 111 554156 110 555180 110 556204 110 557227 111 558251 110 559275 110 560299 109 561322 110 562346 110 563370 109 564394 109 565417 110 566441 110 567465 109 568489 109 569512 110 570536 110 571560 109 572584 109 573608 109 574632 109 575656 108 576679 109 577703 109 578727 109 579751 108 580775 108 581799 108 582823 108 583847 107 584870 108 585894 108 586918 108 587942 107 588966 107 589990 107 591014 107 592038 106 593062 106 594085 106 595109 106 596133 106 597157 105 598181 105 599205 105 600229 104 601253 104 602276 105 603300 104 604324 104 605348 104 606372 103 607396 103 608420 103 609444 102 610468 102 611491 103 612515 103 613539 102 614563 102 615587 102 616611 102 617635 101 618659 101 619683 101 620707 101 621731 100 622755 100 623779 100 624803 100 625827 99 626851 99 627876 98 628900 97 629924 97 630948 97 631972 96 632996 96 634020 96 635044 96 636068 96 637092 95 638116 95 639140 95 640164 95 641188 95 642212 95 643236 95 644260 95 645284 94 646308 94 647332 94 648356 94 649380 94 650404 94 651428 94 652452 94 653476 94 654501 94 655525 94 656549 94 657573 94 658597 94 659622 93 660646 92 661671 91 662695 19 662726 59 663720 15 663754 55 664744 13 664786 47 665769 9 665819 37 666793 1 666852 28 667886 10 730333 1 731353 13 732372 22 733395 27 734418 29 735440 33 736463 35 737486 38 738509 40 739532 41 740556 41 741579 43 742602 44 743626 44 744649 45 745673 46 746696 47 747720 45 748743 44 749767 43 750790 42 751816 32 752843 20 753869 1"
    pneumo_tot = rle_decode(maskToTry, (1024, 1024))
    #pneumo_tot = pn1 + pn2 + pn3 + pn4 + pn5

    #pneumo_tot[pneumo_tot > 1] = 1

    #pneumo_added = pneumo1# + pneumo2

    #test_rle = mask2rle(pneumo_tot)

    #print(f"test rle: {test_rle}")
    #pneumo_tot = rle_decode(test_rle, (1024, 1024))


    # segmentation_mask_org = np.uint8(mask1)
    #mask2 = rle_decode_modified(rib1, (1024, 1024))
    #mask3 = rle_decode_modified(rib2, (1024, 1024))
    #mask4 = rle_decode_modified(rib3, (1024, 1024))
    #mask5 = rle_decode_modified(tube, (1024, 1024))
    #mask_tot = mask1 + mask2 + mask3 + mask4 + (mask5 * 3)

    data_path = "Z:/public_datasets/candid_ptx/dataset1/dataset/"
    img_name = "3.2.87.981173.70.6.2.1.401987205424296.3682398659895.5"
    img_name = "2.8.23.597434.19.0.7.2.05250926090.1548912238334.2"
    img_name = "3.7.30.155952.16.5.6.2.933570534660747.9104634003282.9"
    img_path = os.path.join(data_path, img_name)
    DCM_Img = pdcm.read_file(img_path)
    img_raw = DCM_Img.pixel_array
    img_raw = img_raw * (255 / np.amax(img_raw))  # puts the highest value at 255
    img_raw = np.uint8(img_raw)

    f, ax = plt.subplots(1, 3)
    ax[0].imshow(img_raw, cmap=plt.cm.bone)
    ax[0].set_title('Chest X-ray', size=20)
    ax[1].imshow(pneumo_tot, cmap=plt.cm.bone)
    ax[1].set_title('Physician Segmentation', size=20)
    ax[2].imshow(pneumo_tot, cmap="jet", alpha=1)
    # ax[2].title.set_text('Segmentation on X-Ray', size=10)
    ax[2].set_title('Segmentation on X-Ray', size=20)
    ax[2].imshow(img_raw, cmap=plt.cm.bone, alpha=.5)

    ax[0].axes.xaxis.set_visible(False)
    ax[0].axes.yaxis.set_visible(False)
    ax[1].axes.xaxis.set_visible(False)
    ax[1].axes.yaxis.set_visible(False)
    ax[2].axes.xaxis.set_visible(False)
    ax[2].axes.yaxis.set_visible(False)
    plt.show()
