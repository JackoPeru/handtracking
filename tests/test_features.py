import types
import unittest


def make_hand():
    return [
        types.SimpleNamespace(
            x=(i % 5) * 0.04 + 0.30,
            y=(i // 5) * 0.04 + 0.25,
            z=(i % 3) * 0.01,
        )
        for i in range(21)
    ]


class HandFeatureTests(unittest.TestCase):
    def test_feature_wrapper_preserves_classifier_results(self):
        from handtracking_gestures import (
            HandFeatures,
            fist_fold_metrics,
            grip_class_scores,
            is_fist,
            is_scroll_gesture,
            is_strong_fist,
            normalized_pinch_ratio,
            pointer_other_fingers_valid,
            spock_all_fingers_up,
            spock_pose_score,
            swipe_pose_metrics,
        )

        hand = make_hand()
        features = HandFeatures(hand)
        funcs = (
            grip_class_scores,
            fist_fold_metrics,
            is_fist,
            is_strong_fist,
            is_scroll_gesture,
            pointer_other_fingers_valid,
            swipe_pose_metrics,
            spock_all_fingers_up,
            spock_pose_score,
        )
        for fn in funcs:
            with self.subTest(fn=fn.__name__):
                self.assertEqual(fn(features), fn(hand))
        self.assertEqual(
            normalized_pinch_ratio(features, 8),
            normalized_pinch_ratio(hand, 8),
        )

    def test_repeated_classifiers_reuse_shared_cached_geometry(self):
        from handtracking_gestures import (
            HandFeatures, grip_class_scores, is_fist, is_strong_fist,
        )

        features = HandFeatures(make_hand())
        grip_class_scores(features)
        is_fist(features)
        is_strong_fist(features)

        self.assertIsNotNone(features._grip_scores)
        self.assertIsNotNone(features._gap_ratio)
        self.assertIsNotNone(features._fold_metrics)
        self.assertIsNotNone(features._point_pose)

        before = (
            features._grip_scores,
            features._gap_ratio,
            features._fold_metrics,
            features._point_pose,
        )
        grip_class_scores(features)
        is_fist(features)
        is_strong_fist(features)
        self.assertEqual(before, (
            features._grip_scores,
            features._gap_ratio,
            features._fold_metrics,
            features._point_pose,
        ))

    def test_joint_angles_are_shared_across_grip_open_and_swipe_classifiers(self):
        from handtracking_gestures import (
            HandFeatures, grip_class_scores, is_open_hand, swipe_pose_metrics,
        )

        features = HandFeatures(make_hand())
        grip_class_scores(features)
        after_grip = len(features._angle_cache)
        self.assertGreater(after_grip, 0)

        is_open_hand(features)
        swipe_pose_metrics(features)

        # Open/swipe reuse the same four-finger PIP/DIP angles already used by grip.
        self.assertEqual(len(features._angle_cache), after_grip)


if __name__ == "__main__":
    unittest.main()
