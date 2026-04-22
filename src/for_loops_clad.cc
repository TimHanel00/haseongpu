/**
 * Copyright 2013 Erik Zenker, Carlchristian Eckert, Marius Melzer
 *
 * This file is part of HASEonGPU
 *
 * HASEonGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * HASEonGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with HASEonGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include <for_loops_clad.hpp>
#include <mesh.hpp>
#include <mt19937ar.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <vector>
#define SMALL_FOR_LOOPS 1E-06 // 1�m is considered to be small
#define MAXREFL 5 // number of max amount of reflections <= this has to be done in a better way later (distance defines the max number)
enum RayLife : bool
{
    Alive = true,
    Dead = false,
};
enum class SurfaceKind
{
    Face0,
    Face1,
    Face2,
    Up,
    Down
};

struct CandidateState
{
    SurfaceKind kind;
    int forbiddenTag;
};

struct HitCandidate
{
    double length;
    int nextTri;
    int nextCellZ;
    int nextForbidden;
};

struct PointM
{
    double x;
    double y;
    double z;
    std::optional<unsigned> ptIndex;
    std::optional<unsigned> zIndex;
    [[nodiscard]] double length() const
    {
        return std::sqrt(x * x + y * y + z * z);
    }
    PointM operator/(double scalar) const
    {
        return PointM(x / scalar, y / scalar, z / scalar);
    }
    PointM operator-(PointM const &other) const
    {
        return {x - other.x, y - other.y, z - other.z};
    }
};
struct BaseVersionSerialContext
{
    std::span<const double> p_in;
    std::span<const unsigned> t_in;
    std::span<const double> beta_v;
    std::span<const double> n_x;
    std::span<const double> n_y;
    std::span<const int> neighbors;
    std::span<const float> surface_new;
    std::span<const double> center_x;
    std::span<const double> center_y;
    std::span<const unsigned> n_p;
    std::span<const int> forbidden;
    RayLife ray_life=Alive;   ;
    unsigned nr_points;
    unsigned N_cells;
    unsigned nr_layers;
    int NumRays;
    double z_mesh;
    double sigma_a;
    double sigma_e;
    double N_tot;
};

struct BaseVersionSerial
{
    BaseVersionSerialContext m_context;

    std::vector<double>* m_dndtAse;
    std::vector<double>& m_betaCells;
    std::vector<PointM> m_points;
    float m_hostCrystalFluorescence;

    BaseVersionSerial(
        std::vector<double>* dndtAse,
        unsigned& raysPerSample,
        Mesh& mesh,
        std::vector<double>& betaCells,
        float hostNTot,
        double hostSigmaA,
        double hostSigmaE,
        unsigned hostNumberOfPoints,
        unsigned hostNumberOfTriangles,
        unsigned hostNumberOfLevels,
        float hostThicknessOfPrism,
        float hostCrystalFluorescence)
        : m_dndtAse(dndtAse)
        , m_betaCells(betaCells)
        , m_hostCrystalFluorescence(hostCrystalFluorescence)
    {
        auto p_in = mesh.points.toArray();
        auto t_in = mesh.trianglePointIndices.toArray();
        auto beta_v = mesh.betaVolume.toArray();
        auto normalVecArr = mesh.normalVec.toArray();
        auto centers = mesh.centers.toArray();

        m_context = BaseVersionSerialContext{
            .p_in = p_in,
            .t_in = t_in,
            .beta_v = beta_v,
            .n_x = std::span<const double>(
                normalVecArr.data(),
                hostNumberOfTriangles),
            .n_y = std::span<const double>(
                normalVecArr.data() + hostNumberOfTriangles,
                hostNumberOfTriangles),
            .neighbors = std::span<const int>(
                mesh.triangleNeighbors.toArray().data(),
                3 * hostNumberOfTriangles),
            .surface_new = std::span<const float>(
                mesh.triangleSurfaces.toArray().data(),
                hostNumberOfTriangles),
            .center_x = std::span<const double>(
                centers.data(),
                hostNumberOfTriangles),
            .center_y = std::span<const double>(
                centers.data() + hostNumberOfTriangles,
                hostNumberOfTriangles),
            .n_p = std::span<const unsigned>(
                mesh.triangleNormalPoint.toArray().data(),
                3 * hostNumberOfTriangles),
            .forbidden = std::span<const int>(
                mesh.forbiddenEdge.toArray().data(),
                3 * hostNumberOfTriangles),

            .nr_points = hostNumberOfPoints,
            .N_cells = hostNumberOfTriangles,
            .nr_layers = hostNumberOfLevels,
            .NumRays = static_cast<int>(raysPerSample),
            .z_mesh = hostThicknessOfPrism,
            .sigma_a = hostSigmaA,
            .sigma_e = hostSigmaE,
            .N_tot = hostNTot};
        for (unsigned point_i = 0; point_i < hostNumberOfPoints; ++point_i)
        {
            for (unsigned z = 0; z < m_context.nr_layers; ++z)
            {
                m_points.emplace_back(PointM{m_context.p_in[point_i],m_context.p_in[m_context.nr_points + point_i],z * m_context.z_mesh,std::make_optional(point_i),std::make_optional(z)});
            }
        }
    }

    void operator()()
    {
        mainLoop();
    }

    void mainLoop()
    {
        double u = 0.0, v = 0.0, w = 0.0;
        unsigned t_1, t_2, t_3;
        double p_cx, p_cy, p_cz;
        double x_rand, y_rand, z_rand;
        double gain;

        auto phi = std::vector<double>(m_context.nr_points * m_context.nr_layers, 0.0);

        auto importance = std::vector<double>(m_context.N_cells * (m_context.nr_layers - 1), 0);
        auto N_rays = std::vector<int>(m_context.N_cells * (m_context.nr_layers - 1), 0);

        printf("NumRays: %i\n", m_context.NumRays);
        //for each sample
        for(PointM const &sample_i : m_points)
        {
            if(!sample_i.ptIndex.has_value()||!sample_i.zIndex.has_value())
            {
                throw std::runtime_error("Point initialization failed!");
            }
            int realNumRays = 0;
            calcImportance(sample_i, importance, N_rays, 1);

            //for each triangle in the mesh
            for(unsigned triangle_i = 0; triangle_i < m_context.N_cells; triangle_i++)
            {
                t_1 = m_context.t_in[triangle_i];
                t_2 = m_context.t_in[m_context.N_cells + triangle_i];
                t_3 = m_context.t_in[2 * m_context.N_cells + triangle_i];

                // for each layer k in the prism
                for(unsigned layer_i = 0; layer_i < m_context.nr_layers - 1; layer_i++)
                {
                    realNumRays += N_rays[triangle_i + layer_i * m_context.N_cells];

                    //go over each ray in the prism
                    for(unsigned ray_i = 0; ray_i < N_rays[triangle_i + layer_i * m_context.N_cells]; ray_i++)
                    {
                        // generate the random numbers in the triangle and the z-coordinate
                        PointM randPoint = genRandPoint(t_1, t_2, t_3, layer_i);
                        gain = propagation(randPoint, sample_i, triangle_i, layer_i, 1);

                        phi[sample_i.ptIndex.value() + sample_i.zIndex.value() * (m_context.nr_points)] += gain
                            * m_context.beta_v[triangle_i + layer_i * m_context.N_cells]
                            * importance[triangle_i + layer_i * m_context.N_cells];

                    } // rays loop end
                } // nr_layers loop end
            } // N_cells loop end

            phi[sample_i.ptIndex.value() + sample_i.zIndex.value() * (m_context.nr_points)] = phi[sample_i.ptIndex.value()  + sample_i.zIndex.value() * (m_context.nr_points)] / realNumRays;
        } // nr_points loop end
        for(PointM const &sample_i : m_points)
        {
            unsigned index_i=sample_i.ptIndex.value();
            phi[index_i] = phi[index_i] / (4.0 * 3.14159);
            double gain_local = m_context.N_tot * m_betaCells[index_i] * (m_context.sigma_e + m_context.sigma_a)
                - (m_context.N_tot * m_context.sigma_a);
            m_dndtAse->at(index_i) = gain_local * phi[index_i] / m_hostCrystalFluorescence;
        }

        printf("\ncalculations finished, givig back the data\n");
    }
    //generate a random Point in the given triangle
    PointM genRandPoint(unsigned t_1, unsigned t_2, unsigned t_3, unsigned layer_i) const
    {
        double u = genrand_real3();
        double v = genrand_real3();

        if((u + v) > 1)
        {
            u = 1 - u;
            v = 1 - v;
        }

        double w = 1 - u - v;

        double z_rand = (layer_i + genrand_real3()) * m_context.z_mesh;
        double x_rand = m_context.p_in[t_1] * u + m_context.p_in[t_2] * v + m_context.p_in[t_3] * w;
        double y_rand = m_context.p_in[m_context.nr_points + t_1] * u
            + m_context.p_in[m_context.nr_points + t_2] * v
            + m_context.p_in[m_context.nr_points + t_3] * w;

        return PointM{x_rand, y_rand, z_rand};
    }
    void calcImportance(
        PointM const&p,
        std::vector<double>& importance,
        std::vector<int>& N_rays,
        int N_reflections)
    {
        int i_t, i_z, Rays_dump = 0, rays_left, i_r, rand_t, rand_z;
        double sum_phi = 0.0, surf_tot = 0.0;
        double prop;

        //    calculate the gain from the centers of each of the boxes to the observed point
        //    calculate the gain and make a "mapping"
        //    receipt: pick the point in the center of one cell,
        //    calculate the gain from this point to the observed point,
        //    estimate the inner part of the Phi_ASE - Integral,
        //    scale the amount of rays proportionally with it
        //    sum the amount of rays and scale it to Int=1, which gives the inverse weights
        //    the number of rays is determined via floor(), with ceil(), zero-redions could be added

        //    use the routine "propagation"!, test: no reflections, just exponential

        for(i_t = 0; i_t < static_cast<int>(m_context.N_cells); i_t++)
        {
            for(i_z = 0; i_z < (static_cast<int>(m_context.nr_layers) - 1); i_z++) //remember the definition differences MatLab/C for indices
            {
                auto startPoint=PointM{m_context.center_x[i_t],m_context.center_y[i_t],m_context.z_mesh * (i_z + 0.5)};
                prop = propagation(startPoint,p, i_t, i_z, N_reflections);

                importance[i_t + i_z * m_context.N_cells] = m_context.beta_v[i_t + i_z * m_context.N_cells] * (prop);
                sum_phi += importance[i_t + i_z * m_context.N_cells];
            }
            surf_tot += m_context.surface_new[i_t];
        }

        //    now calculate the number of rays
        for(i_t = 0; i_t < static_cast<int>(m_context.N_cells); i_t++)
        {
            for(i_z = 0; i_z < (static_cast<int>(m_context.nr_layers) - 1); i_z++) //remember the definition differences MatLab/C for indices
            {
                //            this is the amount of the sampled rays out of the cells
                N_rays[i_t + i_z * m_context.N_cells]
                    = (int)(floor(importance[i_t + i_z * m_context.N_cells] / sum_phi * m_context.NumRays));

                Rays_dump += N_rays[i_t + i_z * m_context.N_cells];
            }
        }

        rays_left = m_context.NumRays - Rays_dump;
        //    distribute the remaining not distributed rays randomly
        if((rays_left) > 0)
        {
            for(i_r = 0; i_r < rays_left; i_r++)
            {
                rand_t = (int)(genrand_real3() * m_context.N_cells);
                rand_z = (int)(genrand_real3() * (m_context.nr_layers - 1));
                N_rays[rand_t + rand_z * m_context.N_cells]++;
            }
        }

        //    now think about the mount of rays which would come out of this volume(surface)
        //    dividing this number with the new amount of rays gives the final importance weight for this area!
        for(i_t = 0; i_t < static_cast<int>(m_context.N_cells); i_t++)
        {
            for(i_z = 0; i_z < (static_cast<int>(m_context.nr_layers) - 1); i_z++) //remember the definition differences MatLab/C for indices
            {
                //            this is the amount of the sampled rays out of the cells
                if(N_rays[i_t + i_z * m_context.N_cells] > 0)
                {
                    importance[i_t + i_z * m_context.N_cells]
                        = static_cast<float>(m_context.NumRays) * m_context.surface_new[i_t] / surf_tot / N_rays[i_t + i_z * m_context.N_cells];
                    //                importance[i_t + i_z*N_cells] = NumRays*surface[i_t]/surf_tot;
                }
                else
                {
                    importance[i_t + i_z * m_context.N_cells] = 0; // case of beta of this point == 0 e.g.
                }
            }
        }
    }
    static std::array<CandidateState, 5> createStates()
    {
        return {{
            {SurfaceKind::Face0, 0},
            {SurfaceKind::Face1, 1},
            {SurfaceKind::Face2, 2},
            {SurfaceKind::Up,    3},
            {SurfaceKind::Down,  4},
        }};
    }
    [[nodiscard]] int faceOffset(SurfaceKind kind) const
    {
        switch(kind)
        {
        case SurfaceKind::Face0: return 0;
        case SurfaceKind::Face1: return static_cast<int>(m_context.N_cells);
        case SurfaceKind::Face2: return static_cast<int>(2 * m_context.N_cells);
        default: return 0;
        }
    }
    double propagation(
    PointM const& rand,
    PointM const& p,
    int t_start,
    int mesh_start,
    int N_refl)
    {
        PointM currentPos = rand;
        PointM vec = p - rand;
        double norm = vec.length();
        PointM dir = vec / norm;

        double distance = norm;
        double distance_total = norm;
        double gain = 1.0;

        int tri = t_start;
        int cell_z = mesh_start;
        int forb = -1;

        N_refl = 0;

        while(true)
        {
            double bestLength = distance;
            std::optional<HitCandidate> bestHit;

            for(auto const& state : createStates())
            {
                if(state.forbiddenTag == forb)
                    continue;

                auto hit = tryIntersection(state, currentPos, dir, tri, cell_z, bestLength);
                if(hit && hit->length < bestLength)
                {
                    bestLength = hit->length;
                    bestHit = hit;
                }
            }

            if(!bestHit)
            {
                break;
            }

            gain *= exp(
                m_context.N_tot
                * (m_context.beta_v[tri + cell_z * m_context.N_cells]
                   * (m_context.sigma_e + m_context.sigma_a)
                   - m_context.sigma_a)
                * bestHit->length);

            distance -= bestHit->length;

            currentPos.x += bestHit->length * dir.x;
            currentPos.y += bestHit->length * dir.y;
            currentPos.z += bestHit->length * dir.z;

            if(fabs(distance) < SMALL_FOR_LOOPS)
            {
                m_context.ray_life = Dead;
                break;
            }

            tri = bestHit->nextTri;
            cell_z = bestHit->nextCellZ;
            forb = bestHit->nextForbidden;
        }

        gain /= (distance_total * distance_total);
        return gain;
    }

    [[nodiscard]] std::optional<HitCandidate> tryIntersection(
    CandidateState const& state,
    PointM const& currentPos,
    PointM const& dir,
    int tri,
    int cell_z,
    double currentBestLength) const
{
    double nominator = 0.0;
    double denominator = 0.0;
    double length_help = 0.0;

    switch(state.kind)
    {
    case SurfaceKind::Face0:
    case SurfaceKind::Face1:
    case SurfaceKind::Face2:
    {
        int offset = faceOffset(state.kind);
        int idx = tri + offset;

        nominator =
            (m_context.n_x[idx] * m_context.p_in[m_context.n_p[idx]]
           + m_context.n_y[idx] * m_context.p_in[m_context.n_p[idx] + m_context.nr_points])
          - (m_context.n_x[idx] * currentPos.x + m_context.n_y[idx] * currentPos.y);

        denominator = m_context.n_x[idx] * dir.x + m_context.n_y[idx] * dir.y;

        if(denominator == 0.0)
            return std::nullopt;

        length_help = nominator / denominator;
        if(length_help <= 0.0 || length_help >= currentBestLength)
            return std::nullopt;

        return HitCandidate{
            .length = length_help,
            .nextTri = m_context.neighbors[idx],
            .nextCellZ = cell_z,
            .nextForbidden = m_context.forbidden[idx]
        };
    }

    case SurfaceKind::Up:
    {
        nominator = (cell_z + 1) * m_context.z_mesh - currentPos.z;
        denominator = dir.z;

        if(denominator == 0.0)
            return std::nullopt;

        length_help = nominator / denominator;
        if(length_help <= 0.0 || length_help >= currentBestLength)
            return std::nullopt;

        return HitCandidate{
            .length = length_help,
            .nextTri = tri,
            .nextCellZ = cell_z + 1,
            .nextForbidden = 4
        };
    }

    case SurfaceKind::Down:
    {
        nominator = cell_z * m_context.z_mesh - currentPos.z;
        denominator = dir.z;

        if(denominator == 0.0)
            return std::nullopt;

        length_help = nominator / denominator;
        if(length_help <= 0.0 || length_help >= currentBestLength)
            return std::nullopt;

        return HitCandidate{
            .length = length_help,
            .nextTri = tri,
            .nextCellZ = cell_z - 1,
            .nextForbidden = 3
        };
    }
    }

    return std::nullopt;
}
};
// change 2011/02/23
// added external input of emission and absorption cross section

// change 2011/04/18
// cladding information and calculation added

// change 2011/04/21
// compiler note: mex for_loops_clad.cpp mt19937ar.cpp

//global variables
