#-*- coding: utf-8 -*-

def hsvConverter(color):
    h = float(color[0]) / 2
    s = float(color[1]) / 100 * 255 # Todo: 수정해야함
    v = float(color[2]) / 100 * 255 # Todo: 수정해야함
    return (h, s, v)

def hsvInverter(color):
    h = float(color[0]) * 2
    s = float(color[1]) / 255 * 100
    v = float(color[2]) / 255 * 100
    return (h, s, v)



blueUpper = (217, 100.0, 91.8)

res = hsvConverter(blueUpper)
print res

# res = hsvInverter(a)
# print res



# 255가 만땅인 50을 100으로 환산

# 50을 / 255 100으로 환산이 가능한가??




# HSV =>



