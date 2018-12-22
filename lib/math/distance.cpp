#include "distance.h"

namespace cpt {

Distance Distance::from_m(float m) {
    return Distance(m);
}

Distance Distance::from_mm(float mm) {
    return Distance(mm / 1000.f);
}

}
