

from segmentation_training import segmentation_training
import pandas as pd
import os
from utility import rle_decode, mask2rle, rle_decode_modified
import matplotlib.pyplot as plt
import numpy as np
#from test_model import load_best_model
from contrastive_training import training_loop



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    string = "561281 9 1011 11 1007 14 1003 19 1001 19 999 21 997 25 996 27 994 29 993 29 993 30 993 30 993 30 992 31 992 30 993 30 993 30 994 29 994 29 994 29 994 28 996 27 996 27 996 27 997 26 997 26 997 26 998 25 998 25 999 24 1000 24 999 24 1000 24 999 24 1000 24 1000 23 1000 24 1000 23 1001 23 1000 24 1000 23 1000 24 1000 23 1001 23 1000 24 1000 23 1001 23 1001 23 1000 23 1001 23 1001 22 1002 22 1001 23 1001 22 1002 22 1002 22 1002 22 1002 21 1003 21 1003 21 1003 21 1002 21 1003 21 1003 21 1003 20 1003 21 1003 20 1004 20 1004 20 1003 20 1005 19 1005 19 1005 19 1005 18 1006 18 1006 18 1006 18 1006 18 1006 17 1008 16 1008 16 1008 16 1008 16 1008 16 1008 16 1008 16 1008 16 1009 15 1009 15 1009 15 1009 15 1010 14 1010 14 1010 14 1010 15 1009 15 1010 14 1010 15 1009 15 1009 16 1008 16 1008 16 1008 17 1007 17 1008 16 1008 16 1008 16 1008 16 1009 15 1009 15 1009 16 1008 16 1008 16 1009 15 1010 14 1010 14 1011 13 1012 12 1012 12 1013 11 1014 10 1014 10 1015 9 1015 9 1016 8 1016 8 1017 7 1017 8 1017 7 1018 6 1018 7 1018 6 1019 6 1018 6 1019 5 1020 5 1020 4 1020 4 1021 3 1022 2 1022 2 1023 1"
    string2 = "259379 7 1012 15 1005 18 1002 20 1002 21 1001 22 1000 22 1000 23 1000 23 999 24 999 24 999 23 1000 22 1001 22 1001 22 1000 23 1000 23 1000 22 1000 23 999 24 998 24 998 25 997 26 996 27 996 27 996 27 996 26 997 26 997 26 997 26 996 27 995 27 996 26 997 26 997 26 997 26 997 26 997 25 998 24 999 24 998 24 999 24 999 23 999 23 1000 23 1000 22 1001 22 1001 21 1002 20 1003 19 1005 18 1005 18 1005 18 1005 17 1006 17 1006 17 1006 17 1006 17 1005 18 1005 17 1006 17 1007 16 1007 17 1006 17 1006 18 1005 18 1005 17 1006 17 1006 17 1006 17 1006 17 1006 17 1007 16 1007 17 1006 17 1006 17 1006 17 1006 17 1006 17 1006 17 1006 18 1005 18 1006 17 1006 18 1006 17 1006 18 1006 17 1006 17 1006 18 1006 17 1006 17 1006 18 1006 17 1006 18 1005 18 1006 18 1005 18 1006 18 1005 19 1004 19 1005 19 1004 20 1003 21 1003 20 1004 20 1003 21 1003 21 1002 22 1001 22 1001 23 1000 24 1000 24 999 24 1000 24 999 25 999 25 998 26 998 25 998 26 998 25 998 26 997 26 998 26 997 26 998 26 998 25 998 26 998 25 998 26 997 27 997 26 997 27 997 26 998 26 998 25 999 25 999 25 998 26 998 26 998 25 998 26 998 26 998 26 997 27 997 27 997 27 997 27 997 27 996 27 997 27 997 27 997 27 997 27 997 27 997 27 997 27 997 28 996 28 996 29 994 30 994 31 993 31 993 31 993 31 993 32 992 33 991 33 991 34 990 35 989 36 989 35 989 36 988 37 987 39 985 40 985 40 984 40 984 41 983 42 982 42 982 43 981 43 981 44 980 45 979 45 980 45 979 46 979 46 978 47 978 47 977 47 978 47 977 47 977 48 977 47 977 48 976 49 975 49 975 50 975 49 975 50 975 49 975 50 974 50 975 50 974 51 974 51 973 51 974 51 973 52 973 52 972 53 972 53 971 54 971 53 971 54 971 54 970 55 970 56 969 56 968 56 969 56 969 55 969 56 969 56 969 56 969 57 968 58 967 60 964 64 961 66 959 63 961 57 968 52 974 48 977 45 981 41 985 36 990 31 995 27 1000 23 1004 19 1007 15 1012 11 1015 7 1020 2"
    #string3 = "293157 10000 1021 2 1020 4 1018 6 1016 8 1014 9 1013 11 1011 12 1011 12 1011 13 1010 13 1009 14 1009 14 1009 14 1009 15 1008 15 1008 15 1007 16 1007 16 1007 17 1006 17 1006 17 1007 17 1006 17 1006 17 1006 17 1006 18 1005 18 1006 17 1006 18 1005 18 1006 17 1006 18 1005 18 1005 19 1005 18 1005 19 1004 19 1005 19 1004 19 1004 20 1003 20 1004 19 1004 20 1003 20 1004 20 1003 20 1003 21 1002 21 1003 21 1002 21 1002 22 1002 21 1002 22 1001 22 1001 23 1001 22 1001 23 1000 23 1001 23 1000 23 1000 24 999 24 1000 24 999 24 999 25 999 25 998 25 998 26 998 25 998 26 998 25 998 26 998 25 998 26 998 25 998 26 998 25 998 26 998 26 997 26 997 27 997 26 997 27 997 27 996 27 997 27 996 28 996 27 996 28 996 28 995 28 996 28 995 29 995 28 995 29 994 30 994 29 994 30 994 29 994 30 994 30 993 30 994 30 993 31 993 30 993 31 993 30 993 31 993 31 992 31 993 31 992 32 992 31 992 32 992 31 993 31 992 32 992 31 992 32 992 32 992 31 992 32 992 31 993 31 993 31 992 31 993 31 993 31 993 31 992 31 993 31 993 31 993 31 993 30 994 30 994 30 994 30 994 29 994 30 994 30 994 30 994 29 995 29 995 29 995 29 995 29 995 29 996 28 996 28 996 28 996 28 996 28 996 28 996 28 996 28 996 28 997 27 997 27 997 27 997 27 998 27 997 27 997 28 996 28 997 28 996 28 996 29 996 28 996 29 996 28 996 29 995 30 995 29 995 30 994 30 995 30 994 30 995 30 994 30 994 31 994 30 994 31 994 30 994 31 994 30 994 31 994 31 993 31 994 31 994 30 994 31 994 30 995 30 994 30 995 30 995 29 996 29 995 30 995 29 996 29 995 30 995 29 996 29 996 29 996 28 998 26 999 26 999 25 1000 25 1001 23 1003 21 1005 20 1006 17 1009 15 1011 12 1014 10 1016 8 1019 4 1022 2"
    string4 = "293056 2 1019 5 1018 6 1018 6 1017 7 1016 8 1015 8 1015 9 1014 10 1014 10 1013 10 1014 10 1013 11 1012 12 1012 12 1011 12 1011 13 1011 12 1011 13 1010 13 1010 14 1010 14 1009 15 1008 16 1008 15 1008 16 1008 15 1008 16 1007 16 1008 16 1007 17 1006 18 1006 17 1006 18 1006 17 1006 18 1006 18 1005 18 1006 18 1005 18 1006 18 1005 18 1006 18 1005 19 1005 19 1004 19 1005 19 1004 19 1005 19 1005 18 1006 18 1005 18 1006 18 1006 17 1006 18 1006 18 1005 19 1005 18 1006 18 1006 17 1007 17 1007 16 1008 16 1008 15 1009 15 1008 15 1009 15 1009 14 1010 14 1010 14 1009 14 1010 14 1010 14 1010 14 1010 14 1009 14 1010 14 1010 14 1010 14 1010 14 1009 15 1009 15 1009 15 1009 15 1009 14 1009 15 1009 15 1009 15 1009 15 1009 15 1008 16 1008 16 1008 16 1008 16 1008 15 1009 15 1009 15 1009 16 1008 15 1009 15 1010 14 1010 14 1010 14 1010 13 1011 13 1012 12 1012 12 1012 12 1012 12 1012 12 1013 11 1013 11 1013 11 1013 10 1014 10 1015 9 1015 10 1014 9 1015 9 1015 9 1016 8 1016 8 1017 7 1017 7 1018 6 1018 6 1019 5 1019 5 1020 4 1020 4 1020 5 1019 5 1020 4 1020 5 1020 4 1020 5 1019 5 1020 4 1020 4 1020 5 1019 5 1020 4 1020 4 1021 3 1021 4 1021 4 1020 4 1021 4 1020 5 1019 5 1020 5 1020 4 1021 4 1021 3 1022 3 1023 1 1024 1"
    string5 = "748851 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48 976 48"
    string6 = "827042 42 982 42 982 42 982 42 982 42 982 42 982 42 982 42 982 42 982 42 982 42 982 42 982 42 982 42 982 42 982 42 982 42 982 42 982 42 982 42 982 42 982 42 982 42 982 42 982 42 982 42 982 42"
    #different annotations for same thing
    string7 = "846368 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54"
    string8 = "822718 43 981 43 981 43 981 43 981 43 981 43 981 43 981 43 981 43 981 43 981 43 981 43 981 43 981 43 981 43 981 43 981 43 981 43 981 43 981 43 981 43 981 43 981 43 981 43 981 43 981 43 981 43 981 43 981 43 981 43 981 43 981 43 981 43 981 43 981 43 981 43 981 43 981 43 981 43 981 43 981 43 981 43 981 43 981 43"
    string9 = "807196 59 965 59 965 59 965 59 965 59 965 59 965 59 965 59 965 59 965 59 965 59 965 59 965 59 965 59 965 59 965 59 965 59 965 59 965 59 965 59 965 59 965 59 965 59 965 59 965 59 965 59 965 59 965 59 965 59 965 59 965 59 965 59 965 59 965 59 965 59"
    string10 ="851181 34 990 34 990 34 990 34 990 34 990 34 990 34 990 34 990 34 990 34 990 34 990 34 990 34 990 34 990 34 990 34 990 34 990 34 990 34 990 34 990 34 990 34 990 34 990 34 990 34 990 34 990 34 990 34 990 34"
    string11 ="860435 36 988 36 988 36 988 36 988 36 988 36 988 36 988 36 988 36 988 36 988 36 988 36 988 36 988 36 988 36 988 36 988 36 988 36 988 36 988 36 988 36 988 36 988 36 988 36 988 36 988 36"
    test_string = "250 5 10 5 5 1"
    rle_fracture = "141793 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63 961 63"
    rle_tube = "101141 2 1018 4 1016 4 1016 4 1017 3 1017 4 1016 4 1017 3 1017 4 1016 4 1016 4 1018 2 1020 2 1021 1 1021 2 1020 2 1021 1 1021 2 1021 1 1021 2 1020 2 1021 1 1021 2 1021 1 1021 2 1021 1 1021 2 1020 2 1021 1 1021 2 1021 1 1021 2 1021 1 1021 2 1020 2 1021 1 1021 2 1021 1 1021 2 1020 2 1021 1 1021 2 1021 1 1021 2 1021 1 1021 2 1020 2 1021 1 1021 2 1020 2 1020 2 1020 2 1020 2 1020 2 1020 2 1019 3 1019 2 1020 2 1020 2 1020 2 1019 3 1019 2 1020 2 1020 2 1020 2 1020 2 1019 3 1019 2 1020 2 1020 2 1020 2 1020 2 1019 3 1019 2 1020 2 1020 2 1020 2 1019 3 1019 2 1020 2 1020 2 1020 2 1020 2 1019 3 1018 3 1018 3 1019 2 1019 3 1018 3 1018 3 1018 3 1018 3 1018 3 1019 2 1019 3 1018 3 1018 3 1018 3 1018 3 1018 3 1019 2 1019 3 1018 3 1011 10 998 16 992 16 992 16 998 10 1011 3 1019 2 1019 3 1018 3 1018 3 1019 2 1019 3 1018 3 1018 3 1019 2 1019 3 1018 3 1018 3 1018 3 1019 2 1019 3 1018 3 1018 3 1019 2 1019 3 1018 3 1018 3 1019 2 1019 3 1018 3 1018 3 1019 2 1019 3 1018 3 1018 3 1019 2 1019 3 1018 3 1018 3 1019 2 1019 3 1018 3 1018 3 1018 3 1019 2 1019 3 1018 3 1018 3 1019 2 1019 3 1018 3 1018 3 1019 2 1019 3 1019 2"
    rle_pneumo = "562246 8 9 5 997 31 991 36 987 38 985 40 982 44 978 47 973 52 969 57 966 60 963 64 960 69 206 14 734 79 189 25 730 93 167 36 727 95 149 53 726 96 132 70 725 98 124 77 725 99 119 81 724 100 115 85 723 102 110 89 722 104 105 93 721 105 103 96 719 106 100 98 719 108 97 100 719 108 95 102 718 110 93 103 717 111 92 104 716 112 90 106 716 113 88 107 716 113 87 108 715 114 86 109 715 114 85 109 716 114 84 109 716 116 82 108 718 116 81 108 719 116 79 109 719 117 78 107 722 118 75 105 726 118 73 92 741 118 71 88 747 118 69 86 751 119 66 85 754 119 63 86 756 119 61 86 757 120 58 87 759 121 54 89 760 122 50 90 762 123 46 92 763 124 42 94 764 125 39 94 766 126 35 95 768 128 30 96 770 130 24 98 772 133 14 103 775 222 4 20 778 219 11 12 782 217 807 215 810 212 812 211 813 210 814 208 817 206 818 204 820 203 821 202 823 199 825 198 826 196 828 195 829 194 831 191 833 190 834 188 837 185 839 183 842 180 844 178 846 176 849 173 851 172 852 170 855 167 857 117 3 46 859 112 8 43 861 110 11 41 862 109 14 37 865 108 15 34 867 107 17 31 870 105 19 29 872 104 21 25 874 103 24 22 876 101 27 19 877 100 30 15 880 98 34 9 883 96 38 3 888 94 931 92 932 91 934 88 937 86 938 85 940 82 942 81 944 78 947 76 948 75 950 72 952 71 954 69 956 68 956 67 958 65 960 64 960 63 962 62 963 61 964 60 965 59 966 58 967 57 968 56 969 56 969 55 970 54 971 54 971 53 972 53 972 52 974 50 975 50 975 49 977 47 978 46 979 46 979 45 981 43 982 42 983 42 983 41 985 39 986 39 986 38 988 37 988 36 989 35 991 34 991 34 991 34 991 34 992 33 992 33 992 33 993 32 994 31 995 29 997 28 998 27 999 26 1000 25 1001 24 1002 22 1003 22 1004 21 1004 20 1005 20 1006 19 1006 18 1007 17 1008 16 1009 15 1010 14 1011 13 1012 12 1014 11 1014 8 1017 5 1021 1"

    rib_fract1 = "268763 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50"
    rib_fract2 = "140030 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56 968 56"

    rib3 = "688410 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50 974 50"
    rib4 = "812471 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54 970 54"

    tube1 = "706854 1 1024 2 1024 2 1024 1 1024 2 1024 2 1024 1 1024 2 1024 2 1024 2 1024 1 1024 2 1024 2 1024 1 1024 2 1024 2 1024 1 1024 2 1024 2 1024 1 1024 2 1024 2 1024 2 1024 1 1024 2 1024 2 1024 1 1024 2 1024 2 1024 1 1024 2 1024 2 1024 1 1024 2 1024 2 1024 2 1024 1 1024 2 1024 2 1024 1 1024 2 1024 2 1024 1 1024 2 1024 2 1024 1 1024 2 1024 2 1024 2 1024 1 1024 2 1024 2 1024 1 1024 2 1024 2 1024 1 1024 2 1024 2 1024 1 1024 2 1024 2 1024 2 1024 1 1024 2 1024 2 1024 1 1024 2 1024 2 1024 1 1024 2 1024 2 1024 1 1024 2 1024 2 1024 2 1024 1 1024 2 1024 2 1024 1 1024 2 1024 2 1024 1 1024 2 1024 2 1024 1 1024 2 1024 2 1024 2 1024 1 1024 2 1024 2 1024 1 1024 2 95 3 926 2 88 5 3 2 926 1 82 5 10 1 926 2 75 5 16 1 927 2 69 4 22 2 927 1 63 5 28 1 927 2 56 5 34 1 928 2 49 5 40 2 928 2 42 5 47 1 929 1 36 5 53 2 928 2 29 5 60 1 929 2 23 4 66 1 930 1 17 5 71 2 929 2 10 5 78 1 930 2 3 5 84 2 930 3 91 1 1024 1 1024 2 1024 1 1024 1 1024 2 1024 1 1024 2 1024 1 1024 1 1024 2 1024 1 1024 1 1024 2 1024 1 1024 2 1024 1 1024 1 1024 2 1024 1 1024 2 1024 1 1024 1 1024 2 1024 1 1024 1 1024 2 1024 1 1024 2 1024 1 1024 1 1024 2 1024 1 1024 2 1024 1 1024 1 1024 2 1024 1 1024 1 1024 2 1024 1 1024 2 1024 1 1024 1 1024 2 1024 1 1024 1 1024 2 1024 1 1024 2 1024 1 1024 1 1024 2 1024 1 1024 2 1024 1 1024 1 1024 2 1024 1 1024 1 1024 2 1024 2 1024 3 1024 2 1024 3 1024 3 1024 2 1024 3 1024 3 1024 2 1024 3 1024 3 1024 2 1024 3 1024 3 1024 3 1024 2 1024 3 1024 3 1024 2 1024 3 1024 3 1024 2 1024 3 1024 3 1024 3 1024 2 1024 3 1024 3 1024 2 1024 3 1024 3 1024 2 1024 3 1024 3 1024 2 1024 3 1024 13 1024 24 1024 23 1024 23 1024 24 1024 12"
    tube2 = "635385 1 1023 1 1024 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1023 1 1024 1 1024 1 1024 2 1024 1 1024 1 1024 2 1024 1 1024 1 1024 1 1024 2 1024 1 1024 1 1024 1 1024 2 1024 1 1024 1 1024 2 1024 1 1024 1 1024 1 1024 2 1024 1 1024 1 1024 1 1024 2 1024 1 1024 1 1024 2 1024 1 1024 1 1024 1 1024 2 1024 1 1024 1 1024 2 1024 1 1024 1 1024 1 1024 2 1024 1 1024 1 1024 1 1024 2 1024 1 1024 1 1024 2 1024 1 1024 1 1024 1 1024 2 1024 1 1024 1 1024 1 1024 2 1024 1 1024 1 1024 2 1024 1 1024 1 1024 2 1024 2 1024 2 1024 2 1024 2 1024 2 1024 2 1024 2 1024 2 1024 2 1024 2 1024 2 1024 2 1024 2 1024 2 1024 2 1024 2 1024 2 1024 1 1024 2 1024 2 1024 2 1024 2 1024 2 1024 2 1024 2 1024 2 1024 2 1024 2 1024 2 1024 2 1024 2 1024 2 1024 2 1024 2 1024 2 1024 2 1024 2 1024 2 1024 2 1024 2 1024 3 1024 2 1024 3 1024 2 1024 2 1024 3 1024 2 1024 3 1024 2 1024 2 1024 3 1024 4 1024 4 1024 2"
    #data = np.random.randint(0, 2, (100, 100))
    #data = np.zeros(100,100)
    #for i in range(0,100):
    #    for j in range(0,100):

    #seq = mask2rle(data)
    #print(type(seq))
    print("random sequence")
    #print(seq)



    #test_mask = rle_decode_modified(test_string, (100,100))
    rib_mask1 = rle_decode_modified(rle_pneumo, (1024,1024))
    rib_mask2 = rle_decode_modified(tube2, (1024,1024))
    both = rib_mask1 + rib_mask2
    #7test_mask = rle_decode_modified(rle_pneumo, (1024, 1024))
    #test = plt.imshow(test_mask, cmap=plt.cm.bone)
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(rib_mask1, cmap=plt.cm.bone)
    ax[1].imshow(rib_mask2, cmap=plt.cm.bone)
    ax[2].imshow(both, cmap=plt.cm.bone)

    #plt.show()




    local = False
    if local == True:
        directory_base = "Z:/"
        #directory_base = "/home/zmh001/r-fcb-isilon/research/Bradshaw/"
    else:
        directory_base = "/UserData/"


    #training_loop(seed = 7, batch_size = 32, dir_base= directory_base, epoch = 50, n_classes = 2)

    #load_best_model(dir_base= directory_base)

    seeds = [117, 295, 98, 456, 915, 1367, 712]
    accuracy_list = []

    for seed in seeds:

        acc = segmentation_training(seed = seed, batch_size = 2, dir_base= directory_base, epoch = 15, n_classes = 2)
        acc = 1
        accuracy_list.append(acc)

        matrix = acc
        df = pd.DataFrame(matrix)
        file_name = 'first_vision_run'
        ## save to xlsx file
        filepath = os.path.join(directory_base,
                                '/UserData/Zach_Analysis/result_logs/candid_result/tests/' + str(file_name) +'/confusion_matrix_seed' + str(
                                    seed) + '.xlsx')

        df.to_excel(filepath, index=False)

    print(accuracy_list)

