#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct {
    size_t index;
    double score;
} candidate_t;

static int compare_candidates(const void *left_ptr, const void *right_ptr) {
    const candidate_t *left = (const candidate_t *)left_ptr;
    const candidate_t *right = (const candidate_t *)right_ptr;
    if (left->score > right->score) {
        return -1;
    }
    if (left->score < right->score) {
        return 1;
    }
    if (left->index < right->index) {
        return -1;
    }
    if (left->index > right->index) {
        return 1;
    }
    return 0;
}

static double box_iou(
    const double *left,
    const double *right,
    double offset
) {
    const double intersection_width =
        fmax(0.0, fmin(left[2], right[2]) - fmax(left[0], right[0]) + offset);
    const double intersection_height =
        fmax(0.0, fmin(left[3], right[3]) - fmax(left[1], right[1]) + offset);
    const double intersection = intersection_width * intersection_height;
    const double left_area =
        (left[2] - left[0] + offset) * (left[3] - left[1] + offset);
    const double right_area =
        (right[2] - right[0] + offset) * (right[3] - right[1] + offset);
    const double union_area = left_area + right_area - intersection;
    return union_area > 0.0 ? intersection / union_area : 0.0;
}

size_t nmss_nms(
    const double *boxes,
    const double *scores,
    size_t count,
    double score_threshold,
    double iou_threshold,
    double offset,
    int64_t *output
) {
    if (count == 0) {
        return 0;
    }

    candidate_t *candidates = malloc(count * sizeof(*candidates));
    bool *suppressed = calloc(count, sizeof(*suppressed));
    if (candidates == NULL || suppressed == NULL) {
        free(candidates);
        free(suppressed);
        return SIZE_MAX;
    }

    size_t candidate_count = 0;
    for (size_t index = 0; index < count; ++index) {
        if (scores[index] >= score_threshold) {
            candidates[candidate_count].index = index;
            candidates[candidate_count].score = scores[index];
            ++candidate_count;
        }
    }
    qsort(
        candidates,
        candidate_count,
        sizeof(*candidates),
        compare_candidates
    );

    size_t output_count = 0;
    for (size_t position = 0; position < candidate_count; ++position) {
        if (suppressed[position]) {
            continue;
        }
        const size_t current_index = candidates[position].index;
        output[output_count++] = (int64_t)current_index;
        const double *current_box = boxes + current_index * 4;

        for (size_t other = position + 1; other < candidate_count; ++other) {
            if (suppressed[other]) {
                continue;
            }
            const size_t other_index = candidates[other].index;
            if (
                box_iou(current_box, boxes + other_index * 4, offset)
                > iou_threshold
            ) {
                suppressed[other] = true;
            }
        }
    }

    free(candidates);
    free(suppressed);
    return output_count;
}
