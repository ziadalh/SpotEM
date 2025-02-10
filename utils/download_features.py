import argparse
import os
import os.path as osp
import subprocess as sp

spotem_root = os.environ["SPOTEM_ROOT"]


DOWNLOAD_URLS = {
    "clip": "https://utexas.box.com/shared/static/lzynfv2hgpk432cynle1egge9wirnw4m.zip",
    "egovlp": "https://utexas.box.com/shared/static/ixinsqrbscbnbpmeg1ac5ebyz16rz9km.zip",
    "egovlp+imagenet": [
        "https://utexas.box.com/shared/static/23yizwgvpzelcdjiggin4j5uzc92klfi.zip",
        "https://utexas.box.com/shared/static/7jtxywuilrqv9vwfnpch1elqtt874xkl.z01",
        "https://utexas.box.com/shared/static/hyp5yxal2lhmb505w84lfoopuiqrts7i.z02",
        "https://utexas.box.com/shared/static/7hk4xtou7fyvrflgx7c9qqd1iolzsgwk.z03",
    ],
    "egovlp+rio": [
        "https://utexas.box.com/shared/static/9yd3dk978iqxnzuoxa8k65ny3w5fqibo.zip",
        "https://utexas.box.com/shared/static/ia7esxjem3mb5r1srf8pc17kq3jys31d.z01",
    ],
    "internvideo": "https://utexas.box.com/shared/static/oahb4kbsgnru8w2rlgunir75nrlaj1no.zip",
    "internvideo+imagenet": [
        "https://utexas.box.com/shared/static/e4mni5fe4nn36lm95xqexdhbqsnhmift.zip",
        "https://utexas.box.com/shared/static/gfuttdn5x3614t3divlkddhz34wnhdo7.z01",
    ],
    "internvideo+rio": [
        "https://utexas.box.com/shared/static/8at3tvk2omg0rencche79j3kantz8b0a.zip",
        "https://utexas.box.com/shared/static/kzpn2nvi05vldst6bqfk5h1x6ai31svi.z01",
    ],
    "room": "https://utexas.box.com/shared/static/ufbubwp5qglxts26ommg5uswnvdykxrs.zip",
    "interaction": "https://utexas.box.com/shared/static/lre8a49j64l3r846r17jldyhn1vtv0wb.zip",
    "object": "https://utexas.box.com/shared/static/cuwpohtulyuaokn52kz6zcgr6s0ua2is.zip",
    "imagenet": [
        "https://utexas.box.com/shared/static/58c0kggl2tonspujtn0hfkogez0p382i.zip",
        "https://utexas.box.com/shared/static/7khmvnp0k23ywh0tq4l7vnodhcftku7u.z01",
        "https://utexas.box.com/shared/static/6arvi5t9toa9140kwcyu3f74do770jjf.z02",
        "https://utexas.box.com/shared/static/3p94ncxzpf33z9wmnia5m9zvi10fc0nm.z03",
    ],
}


def download_features(feature_type):
    assert (
        feature_type in DOWNLOAD_URLS
    ), f"{feature_type} not in {DOWNLOAD_URLS.keys()}"
    os.makedirs(osp.join(spotem_root, "data/features/nlq_official_v1"), exist_ok=True)
    print(f"===> Downloading {feature_type} features")
    urls = DOWNLOAD_URLS[feature_type]
    if type(urls) is str:
        destination = osp.join(
            spotem_root, f"data/features/nlq_official_v1/{feature_type}.zip"
        )
        sp.call(
            [
                "wget",
                "-nv",
                "--show-progress",
                "-O",
                destination,
                DOWNLOAD_URLS[feature_type],
            ]
        )
    else:
        # Download split zip files
        for idx, url in enumerate(urls):
            if idx == 0:
                destination = osp.join(
                    spotem_root,
                    f"data/features/nlq_official_v1/{feature_type}_split.zip",
                )
            else:
                destination = osp.join(
                    spotem_root,
                    f"data/features/nlq_official_v1/{feature_type}_split.z{idx:02d}",
                )
            sp.call(["wget", "-nv", "--show-progress", "-O", destination, url])
        # Merge zip files into one
        print("Merging split zip files. Might take a while...")
        sp.call(
            [
                "zip",
                "-q",
                "-s",
                "0",
                osp.join(
                    spotem_root,
                    f"data/features/nlq_official_v1/{feature_type}_split.zip",
                ),
                "--out",
                osp.join(
                    spotem_root, f"data/features/nlq_official_v1/{feature_type}.zip"
                ),
            ]
        )
        # Delete split zip files
        for idx in range(len(urls)):
            if idx == 0:
                destination = osp.join(
                    spotem_root,
                    f"data/features/nlq_official_v1/{feature_type}_split.zip",
                )
            else:
                destination = osp.join(
                    spotem_root,
                    f"data/features/nlq_official_v1/{feature_type}_split.z{idx:02d}",
                )
            sp.call(["rm", destination])

    print("Extracting features...")
    sp.call(
        [
            "unzip",
            "-qq",
            "-d",
            osp.join(spotem_root, "data/features/nlq_official_v1"),
            osp.join(spotem_root, f"data/features/nlq_official_v1/{feature_type}.zip"),
        ]
    )

    # Delete zip file
    sp.call(
        [
            "rm",
            osp.join(spotem_root, f"data/features/nlq_official_v1/{feature_type}.zip"),
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature_types",
        nargs="+",
        type=str,
        default="egovlp",
        choices=list(DOWNLOAD_URLS.keys()),
    )
    args = parser.parse_args()

    for feature_type in args.feature_types:
        download_features(feature_type)
